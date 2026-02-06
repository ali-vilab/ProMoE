import os
import os.path as osp
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
import numpy as np
import logging
import datetime
from PIL import Image
from config import cfg
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import argparse
import yaml
import colorlog
from diffusers.models import AutoencoderKL
from train import model_dict
import glob
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import math
import inspect
from concurrent.futures import ThreadPoolExecutor
from utils import InceptionV3, deep_update, find_free_port, str_to_float_list, str_to_int_list

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_logging(output_dir, rank):
    os.makedirs(output_dir, exist_ok=True)
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s-%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    if rank == 0:
        file_handler = logging.FileHandler(os.path.join(output_dir, "sample.log"))
        file_handler.setFormatter(formatter)  # Using colorlog is not effective in files but format can still apply
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)


def load_specific_checkpoints(checkpoint_dir, specific_steps):
    # Ensure specific_steps is a set for faster lookups
    specific_steps = set(specific_steps)
    # Find all checkpoint files
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'ckpt_step_*.pth'))

    specific_checkpoints = []
    step_to_checkpoint = {}
    for checkpoint in all_checkpoints:
        step_str = os.path.basename(checkpoint).split('_')[-1].replace('.pth', '')
        try:
            step = int(step_str)
            if step in specific_steps:
                specific_checkpoints.append(checkpoint)
                step_to_checkpoint[checkpoint] = step
        except ValueError:
            continue

    if not specific_checkpoints:
        logging.info(f'No checkpoints found for specified steps: {specific_steps}')
        return [], []

    # Sort checkpoints and steps by step
    sorted_pairs = sorted(specific_checkpoints, key=lambda x: step_to_checkpoint[x])
    sorted_checkpoints = [checkpoint for checkpoint in sorted_pairs]
    sorted_steps = [step_to_checkpoint[checkpoint] for checkpoint in sorted_pairs]

    logging.info(f'Found {len(sorted_checkpoints)} checkpoints for specified steps: {specific_steps}')
    return sorted_checkpoints, sorted_steps

def load_checkpoints_every_val_step(checkpoint_dir, sample_every_step):
    # Find all checkpoint files
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'ckpt_step_*.pth'))

    specific_checkpoints = []
    step_to_checkpoint = {}
    for checkpoint in all_checkpoints:
        step_str = os.path.basename(checkpoint).split('_')[-1].replace('.pth', '')
        try:
            step = int(step_str)
            if step % sample_every_step == 0:
                specific_checkpoints.append(checkpoint)
                step_to_checkpoint[checkpoint] = step
        except ValueError:
            continue
    
    if not specific_checkpoints:
        logging.info(f'No checkpoints found for step multiples of {sample_every_step}')
        return [], []

    # Sort checkpoints and steps by step
    sorted_pairs = sorted(specific_checkpoints, key=lambda x: step_to_checkpoint[x])
    sorted_checkpoints = [checkpoint for checkpoint in sorted_pairs]
    sorted_steps = [step_to_checkpoint[checkpoint] for checkpoint in sorted_pairs]

    logging.info(f'Found {len(sorted_checkpoints)} checkpoints for step multiples of {sample_every_step}')
    return sorted_checkpoints, sorted_steps


def get_sampling_sigmas(sampling_steps, shift):
    # extra step for zero
    # sigma = torch.linspace(1, 0, sampling_steps + 1)
    sigma = np.linspace(1, 0, sampling_steps+1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps= None,
    device= None,
    timesteps= None,
    sigmas = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def main(**kwargs):
    deep_update(cfg, kwargs)

    if 'sample_gpu_ids' in kwargs and kwargs['sample_gpu_ids'] is not None:
        sample_gpu_ids = ','.join(map(str, kwargs['sample_gpu_ids']))
        os.environ['CUDA_VISIBLE_DEVICES'] = sample_gpu_ids
        print(f"Set CUDA_VISIBLE_DEVICES to {sample_gpu_ids}")

    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']= find_free_port()

    cfg.output_dir = osp.join(cfg.output_dir, cfg.model_name, cfg.custom_cfg_name)

    cfg.pmi_rank = int(os.getenv('RANK', 0))
    cfg.pmi_world_size = 1

    if 'sample_gpu_ids' in kwargs and kwargs['sample_gpu_ids'] is not None:
        cfg.gpus_per_machine = len(kwargs['sample_gpu_ids'])
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    print(f'cfg.world_size: {cfg.world_size}')
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg

@torch.no_grad
def worker(gpu, cfg):
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    cfg.seed = cfg.global_seed * cfg.pmi_world_size + cfg.rank
    torch.manual_seed(cfg.seed)

    setup_logging(cfg.output_dir, cfg.rank)

    # init distributed processes
    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        rank=cfg.rank,
        world_size=cfg.world_size,
        timeout=datetime.timedelta(hours=5)
    )

    logging.info('Initializing VAE, Inception')

    # [model] vae
    vae = AutoencoderKL.from_pretrained(cfg.sd_vae_ft_mse_vae_path)  # [B, 16, 1, 32, 32] img 256x256
    vae = vae.eval().to(gpu)
    latent_shape = (4, 1, cfg.image_size // 8, cfg.image_size // 8)

    cfg.save_inception_features = getattr(cfg, 'save_inception_features', False)
    if cfg.save_inception_features:
        inception = InceptionV3().to(gpu).eval()

    cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoints')
    if hasattr(cfg, 'step_list_for_sample') and cfg.step_list_for_sample:
        cfg.val_loss_model, cfg.val_loss_model_steps = load_specific_checkpoints(cfg.checkpoint_dir, cfg.step_list_for_sample)
    else:
        cfg.val_loss_model, cfg.val_loss_model_steps = load_checkpoints_every_val_step(cfg.checkpoint_dir, cfg.sample_every_step)
    
    if hasattr(cfg, 'guide_scale_list') and cfg.guide_scale_list:
        guide_scales = cfg.guide_scale_list
    else:
        guide_scales = [cfg.guide_scale]
    
    for ckpt_path, ckpt_step in zip(cfg.val_loss_model, cfg.val_loss_model_steps):
        for current_guide_scale in guide_scales:
            folder_name = f"img{cfg.image_size}_cfg{current_guide_scale}_seed{cfg.global_seed}_FID{int(cfg.num_fid_samples/1000)}K_bs{cfg.sample_batch_size}_ema"
            cfg.sample_folder_dir = osp.join(cfg.output_dir, 'sample', f'step{ckpt_step}', folder_name)
            os.makedirs(cfg.sample_folder_dir, exist_ok=True)
            logging.info(f"Saving .png samples at {cfg.sample_folder_dir} with guide_scale={current_guide_scale}")
            cfg.sample_images_folder_dir = osp.join(cfg.sample_folder_dir, 'images')
            os.makedirs(cfg.sample_images_folder_dir, exist_ok=True)
            if cfg.save_inception_features:
                cfg.sample_inception_features_folder_dir = osp.join(cfg.sample_folder_dir, 'inception_features')
                os.makedirs(cfg.sample_inception_features_folder_dir, exist_ok=True)
            
            n = cfg.sample_batch_size
            global_batch_size = n * cfg.world_size
            total_samples = int(math.ceil(cfg.num_fid_samples / global_batch_size) * global_batch_size)
            assert total_samples % cfg.world_size == 0, "total_samples must be divisible by world_size"
            ori_total_samples = total_samples

            samples_per_gpu = total_samples // cfg.world_size
            iterations = samples_per_gpu // n
            
            # [model] transformer
            logging.info('Initializing transformer models (non-ema and ema)')
            model_class, config_name = model_dict[cfg.model_name]
            model_cfg = getattr(cfg, config_name)
            logging.info(f'model_cfg: {model_cfg}')
            model = model_class(**model_cfg)

            checkpoint = torch.load(ckpt_path, map_location='cpu')
            missing_key, unexpected_key = model.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)
            logging.info(f"missing key: {missing_key}")
            logging.info(f"unexpected key: {unexpected_key}")

            model = model.to(gpu)
            model = DistributedDataParallel(model, device_ids=[gpu])

            model_size = sum([p.numel() for p in model.parameters()]) / (1000 ** 3)
            logging.info(f'Created models with {model_size:.3f} billion parameters')
            torch.cuda.empty_cache()
            
            logging.info('Start the sample loop')
            model_val = model.eval()
            pbar = range(iterations)
            pbar = tqdm(pbar) if cfg.rank == 0 else pbar
            save_executor = ThreadPoolExecutor(max_workers=8)

            for i in pbar:
                noise = torch.randn(n, *latent_shape, device=gpu)
                y = torch.randint(0, cfg.num_classes, (n,), device=gpu)
                y_null = torch.tensor([cfg.num_classes] * n, device=gpu)

                global_index = i * cfg.world_size + cfg.rank
                
                if cfg.save_inception_features:
                    inception_file_path = os.path.join(cfg.sample_inception_features_folder_dir, f"{global_index:06d}.npy")
                    if os.path.exists(inception_file_path):
                        if cfg.rank == 0:
                            logging.info(f"Skipping batch with global_index {global_index} because inception feature file exists.")
                        continue
                else:
                    batch_complete = True
                    for img_idx in range(n):
                        image_pattern = os.path.join(cfg.sample_images_folder_dir, f"img{global_index * n + img_idx:06d}_class*.png")
                        matching_files = glob.glob(image_pattern)
                        if not matching_files:
                            batch_complete = False
                            break
                    
                    if batch_complete:
                        if cfg.rank == 0:
                            logging.info(f"Skipping batch with global_index {global_index} because all image files exist.")
                        continue
                
                with amp.autocast(dtype=cfg.val_param_dtype):
                    sample_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=cfg.shift)
                    sampling_sigmas = get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift)

                    latents = noise

                    timesteps, num_inference_steps = retrieve_timesteps(sample_scheduler, device=gpu, sigmas=sampling_sigmas)
                    arg_c = {'context': y, 'use_gradient_checkpointing': cfg.use_gradient_checkpointing}
                    arg_null = {'context': y_null, 'use_gradient_checkpointing': cfg.use_gradient_checkpointing}

                    for i_t, t in enumerate(timesteps):
                        latent_model_input = latents
                        timestep = [t] * len(latents)
                        timestep = torch.stack(timestep)

                        noise_pred_cond = model_val(latent_model_input, timestep, **arg_c)
                        if isinstance(noise_pred_cond, tuple):
                            noise_pred_cond = noise_pred_cond[0]

                        if current_guide_scale > 1.0:
                            noise_pred_uncond = model_val(latent_model_input, timestep, **arg_null)
                            if isinstance(noise_pred_uncond, tuple):
                                noise_pred_uncond = noise_pred_uncond[0]
                            noise_pred = noise_pred_uncond + current_guide_scale * (noise_pred_cond - noise_pred_uncond)
                        else:
                            noise_pred = noise_pred_cond
                        
                        if noise_pred.shape[1] != latents.shape[1]:
                            noise_pred, _ = noise_pred.chunk(2, dim=1)

                        latents = sample_scheduler.step(noise_pred.unsqueeze(2), t, latents, return_dict=False)[0]
                    x0 = latents
                
                samples = vae.decode(x0.squeeze(2) / 0.18215).sample
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255)

                for img_idx in range(samples.size(0)):
                    sample = samples[img_idx]
                    class_label = y[img_idx].item()
                    sample = sample.cpu()
                    sample = sample.permute(1, 2, 0).numpy().astype(np.uint8)
                    image_filename = f"img{global_index * n + img_idx:06d}_class{class_label}.png"
                    image_path = os.path.join(cfg.sample_images_folder_dir, image_filename)
            
                    def save_image(img_array, path):
                        img = Image.fromarray(img_array)
                        img.save(path)
                    
                    save_executor.submit(save_image, sample, image_path)

                if cfg.save_inception_features:
                    inception_feature = inception(samples / 255.).cpu().numpy()
                    
                    def save_inception_feature(feature, path):
                        np.save(path, feature)
                    
                    save_executor.submit(save_inception_feature, inception_feature, inception_file_path)

            save_executor.shutdown(wait=True)
            dist.barrier()

            if cfg.rank == 0:
                if cfg.save_inception_features:
                    def get_all_filenames_in_folder(folder_path):
                        if not os.path.isdir(folder_path):
                            print(f"Error: {folder_path} is an illegal path. ")
                            return []
                        filenames = os.listdir(folder_path)
                        return filenames
                    sample_dir = cfg.sample_inception_features_folder_dir + '/'
                    filenames = get_all_filenames_in_folder(sample_dir)

                    def create_npz_from_sample_folder(sample_dir):
                        activations = []
                        cnt = 0
                        for name in tqdm(filenames):
                            feature = np.load(sample_dir+name)
                            activations.append(feature)
                            cnt += 1

                        activations = np.concatenate(activations)
                        print(activations.shape)
                        npz_path = f"{cfg.sample_folder_dir}/{folder_name}.npz"
                        mu = np.mean(activations, axis=0)
                        sigma = np.cov(activations, rowvar=False)
                        np.savez(npz_path, activations=activations, mu=mu, sigma=sigma)
                        print(f"Saved .npz file to {npz_path} [shape={activations.shape}].")
                        return npz_path
                    logging.info(filenames)
                    create_npz_from_sample_folder(sample_dir, num=ori_total_samples)
                    logging.info("Done.")
    
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample for MoE')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--step_list_for_sample', 
                        type=str_to_int_list, 
                        default=None, 
                        help='Comma-separated list of integers to override step_for_sample, e.g., "100,200,300"')
    parser.add_argument('--guide_scale_list', 
                        type=str_to_float_list, 
                        default=None, 
                        help='Comma-separated list of floats to override guide_scale, e.g., "1.0,1.5"')
    parser.add_argument('--num_fid_samples',
                        type=int,
                        default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        custom_cfg = yaml.safe_load(file)
    
    custom_cfg['custom_cfg_name'] = osp.splitext(osp.basename(args.config))[0]
    
    if args.step_list_for_sample is not None:
        print(f"Overriding 'step_list_for_sample' from config with command-line value: {args.step_list_for_sample}")
        custom_cfg['step_list_for_sample'] = args.step_list_for_sample

    if args.guide_scale_list is not None:
        print(f"Overriding 'guide_scale_list' from config with command-line value: {args.guide_scale_list}")
        custom_cfg['guide_scale_list'] = args.guide_scale_list
    
    if args.num_fid_samples is not None:
        print(f"Setting num_fid_samples from command-line: {args.num_fid_samples}")
        custom_cfg['num_fid_samples'] = args.num_fid_samples
        custom_cfg['save_img_num'] = args.num_fid_samples

    main(**custom_cfg)