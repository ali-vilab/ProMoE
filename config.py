import torch
from easydict import EasyDict
import os

cfg = EasyDict(__name__='Config: MoE')

# -------------------------------distributed training--------------------------
pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
gpus_per_machine = torch.cuda.device_count()
world_size = pmi_world_size * gpus_per_machine
# -----------------------------------------------------------------------------


# ---------------------------Dataset Parameter---------------------------------
cfg.num_classes = 1000
cfg.data_path = "/path/to/ImageNet/train"

cfg.img_num_workers = 8
cfg.prefetch_factor = 2

cfg.preprocess_batch_size = 256
cfg.image_size = 256

# -----------------------------------------------------------------------------


# ---------------------------Mode Parameters-----------------------------------
cfg.vae_type = "sd-vae-ft-mse"
cfg.sd_vae_ft_mse_vae_path = "stabilityai/sd-vae-ft-mse"

# ----------------- DiT model
cfg.DiT_S_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 12,
    'hidden_size': 384,
    'num_heads': 6,
    'mlp_ratio': 4,
    'use_swiglu': False,
}

cfg.DiT_B_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 12,
    'hidden_size': 768,
    'num_heads': 12,
    'mlp_ratio': 4,
    'use_swiglu': False,
}

cfg.DiT_M_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 16,
    'hidden_size': 960,
    'num_heads': 16,
    'mlp_ratio': 4,
    'use_swiglu': False,
}

cfg.DiT_L_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 24,
    'hidden_size': 1024,
    'num_heads': 16,
    'mlp_ratio': 4,
    'use_swiglu': False,
}

cfg.DiT_XL_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 28,
    'hidden_size': 1152,
    'num_heads': 16,
    'mlp_ratio': 4,
    'use_swiglu': False,
}


# ----------------- MoE model
cfg.DiffMoE_DiT_B_E8_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 12,
    'hidden_size': 768,
    'num_heads': 12,
    'mlp_ratio': 4,
    'use_swiglu': False,
    'MoE_config': {
        'n_shared_experts': 0,
        'num_experts': 8,
        'capacity': 1,
        'init_MoeMLP': False,
        'interleave': True,
        'CapacityPred_loss_weight': 1
    }
}

cfg.DiffMoE_DiT_L_E8_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 24,
    'hidden_size': 1024,
    'num_heads': 16,
    'mlp_ratio': 4,
    'use_swiglu': False,
    'MoE_config': {
        'n_shared_experts': 0,
        'num_experts': 8,
        'capacity': 1,
        'init_MoeMLP': False,
        'interleave': True,
        'CapacityPred_loss_weight': 1
    }
}

cfg.DiffMoE_DiT_XL_E8_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 28,
    'hidden_size': 1152,
    'num_heads': 16,
    'mlp_ratio': 4,
    'use_swiglu': False,
    'MoE_config': {
        'n_shared_experts': 0,
        'num_experts': 8,
        'capacity': 1,
        'init_MoeMLP': False,
        'interleave': True,
        'CapacityPred_loss_weight': 1
    }
}

cfg.TCDiT_L_E8_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 24,
    'hidden_size': 1024,
    'num_heads': 16,
    'mlp_ratio': 4,
    'use_swiglu': False,
    'MoE_config': {
        'n_shared_experts': 0,
        'num_experts': 8,
        'capacity': 1,
        'init_MoeMLP': False,
        'interleave': True,
    }
}

cfg.ECDiT_L_E8_config = {
    'input_size': 32,
    'num_classes': cfg.num_classes,
    'patch_size': 2,
    'depth': 24,
    'hidden_size': 1024,
    'num_heads': 16,
    'mlp_ratio': 4,
    'use_swiglu': False,
    'MoE_config': {
        'n_shared_experts': 0,
        'num_experts': 8,
        'capacity': 1,
        'init_MoeMLP': False,
        'interleave': True,
    }
}
# -----------------------------------------------------------------------------

# ---------------------------Training Settings---------------------------------
### train
cfg.use_pre_latents = False
cfg.resume_checkpoint = False
cfg.use_gradient_checkpointing = False

cfg.grad_mix = 1
cfg.num_steps = 10_000_000
cfg.betas = (0.9, 0.999)

cfg.num_train_timesteps = 1000
cfg.shift = 1.0
cfg.sigmoid_scale = 1.0

cfg.weighting_scheme = 'logit_normal'
cfg.logit_mean = 0.0
cfg.logit_std = 1.0
cfg.mode_scale = 1.29
cfg.max_grad_norm = 0.5

cfg.param_dtype = torch.bfloat16

cfg.save_ckpt_interval = 1000
cfg.output_dir = "outputs/"
cfg.log_interval = 10

### sample
cfg.val_param_dtype = torch.float32
cfg.global_seed = 0
# -----------------------------------------------------------------------------