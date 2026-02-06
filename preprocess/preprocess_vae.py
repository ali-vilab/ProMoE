import os
import os.path as osp
import sys
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import logging
import datetime
from torch.utils.data import Dataset, DataLoader
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)
from torchvision import transforms
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL
from config import cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CustomImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.CACHE_FILE = 'preprocess/image_paths_cache.txt'
        self.image_paths = self._load_or_generate_image_paths()

    def _load_or_generate_image_paths(self):
        if os.path.exists(self.CACHE_FILE) and os.path.getsize(self.CACHE_FILE) > 0:
            with open(self.CACHE_FILE, 'r') as f:
                image_paths = f.read().splitlines()
            logging.info(f"Loaded image paths from cache: {self.CACHE_FILE}")
            return image_paths
        
        image_paths = self._get_image_paths(self.root_dir)
        
        # Save to cache for future use
        with open(self.CACHE_FILE, 'w') as f:
            f.write('\n'.join(image_paths))
        
        logging.info(f"Generated cache for image paths: {self.CACHE_FILE}")
        return image_paths

    def _get_image_paths(self, root_dir):
        image_paths = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = []
            for entry in os.scandir(root_dir):
                if entry.is_dir(follow_symlinks=False):
                    futures.append(executor.submit(self._get_image_paths_from_dir, entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(entry.path)

            for future in as_completed(futures):
                image_paths.extend(future.result())
        
        return image_paths

    def _get_image_paths_from_dir(self, dir_path):
        image_paths = []
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(entry.path)
                elif entry.is_dir(follow_symlinks=False):
                    image_paths.extend(self._get_image_paths_from_dir(entry.path))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  
        if self.transform:
            image = self.transform(image)

        return image, img_path


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def main(**kwargs):
    cfg.update(**kwargs)

    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()

    cfg.pmi_rank = int(os.getenv('RANK', 0))
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    print(f'Used GPU Number: {cfg.world_size}')
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg,))
    return cfg

def save_latent(encoded_latent, encoded_latent_flipped, image_path, save_root):
    relative_path = osp.relpath(image_path, cfg.data_path)
    relative_path = osp.splitext(relative_path)[0] + '.latent.npz'
    latent_save_path = osp.join(save_root, relative_path)
    latent_dir = osp.dirname(latent_save_path)
    os.makedirs(latent_dir, exist_ok=True)
    latent_np = encoded_latent.detach().cpu().numpy()
    latent_flip_np = encoded_latent_flipped.detach().cpu().numpy()
    # Save both original and flipped latent numpy arrays as a compressed `.npz` file
    np.savez_compressed(latent_save_path, latent=latent_np, latent_flip=latent_flip_np)


@torch.no_grad
def worker(gpu, cfg):
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu

    # init distributed processes
    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        rank=cfg.rank,
        world_size=cfg.world_size,
        timeout=datetime.timedelta(hours=5)
    )

    transform = transforms.Compose([
        transforms.Lambda(center_crop_lambda),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    img_dataset = CustomImageDataset(cfg.data_path, transform=transform)
    sampler = DistributedSampler(img_dataset, num_replicas=cfg.world_size, rank=cfg.rank)

    # Define Sequential Dataloader (default behavior without sampler)
    image_dataloader = DataLoader(
        img_dataset,
        batch_size=cfg.preprocess_batch_size,
        sampler=sampler,  # Ensure the data is processed in a distributed manner
        num_workers=cfg.img_num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor
    )

    latent_save_root = cfg.latent_save_root

    logging.info('Start the preprocess loop')

    # Initialize VAE
    vae = AutoencoderKL.from_pretrained(cfg.sd_vae_ft_mse_vae_path)
    vae = vae.eval().to(gpu)

    # Create ThreadPoolExecutor for saving latents
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for batch_idx, (rank_images, rank_image_paths) in tqdm(enumerate(image_dataloader), total=len(image_dataloader)):
            rank_images = rank_images.to(gpu, non_blocking=True)
            
            rank_img_z = vae.encode(rank_images).latent_dist.parameters
            rank_img_z_flipped = vae.encode(rank_images.flip(dims=[3])).latent_dist.parameters

            # Submit saving tasks to the executor
            for idx in range(len(rank_images)):
                img_path = rank_image_paths[idx]
                latent = rank_img_z[idx]
                latent_flipped = rank_img_z_flipped[idx]
                future = executor.submit(save_latent, latent, latent_flipped, img_path, latent_save_root)
                futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Ensure all futures are completed

    if cfg.rank == 0:
        logging.info('Congratulations! The preprocess is completed!')
    
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def center_crop_lambda(pil_image):
    return center_crop_arr(pil_image, cfg.image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run latent extraction/processing script.")
    parser.add_argument('--latent_save_root', type=str, required=True, help="Root directory path to save or load the latents")
    args = parser.parse_args()

    main(latent_save_root=args.latent_save_root)
