
# [ICLR 2026] Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.24711-b31b1b.svg)](https://arxiv.org/abs/2510.24711)

_**[Yujie Wei<sup>1</sup>](https://weilllllls.github.io), [Shiwei Zhang<sup>2*</sup>](https://scholar.google.com.hk/citations?user=ZO3OQ-8AAAAJ), [Hangjie Yuan<sup>3</sup>](https://jacobyuan7.github.io), [Yujin Han<sup>4</sup>](https://yujinhanml.github.io/), [Zhekai Chen<sup>4,5</sup>](https://scholar.google.com/citations?user=_eZWcIMAAAAJ), [Jiayu Wang<sup>2</sup>](https://openreview.net/profile?id=~Jiayu_Wang2), [Difan Zou<sup>4</sup>](https://difanzou.github.io/), [Xihui Liu<sup>4,5</sup>](https://xh-liu.github.io/), [Yingya Zhang<sup>2</sup>](https://scholar.google.com/citations?user=16RDSEUAAAAJ), [Yu Liu<sup>2</sup>](https://scholar.google.com/citations?user=8zksQb4AAAAJ), [Hongming Shan<sup>1†</sup>](http://hmshan.io)**_
<br>
(*Project Leader, †Corresponding Author)

<sup>1</sup>Fudan University <sup>2</sup>Tongyi Lab, Alibaba Group <sup>3</sup>Zhejiang University <sup>4</sup>The University of Hong Kong <sup>5</sup>MMLab
</div>

Mixture-of-Experts (MoE) has emerged as a powerful paradigm for scaling model capacity while preserving computational efficiency. Despite its notable success in large language models (LLMs), existing attempts to apply MoE to Diffusion Transformers (DiTs) have yielded limited gains. We attribute this gap to fundamental differences between language and visual tokens. Language tokens are semantically dense with pronounced inter-token variation, while visual tokens exhibit spatial redundancy and functional heterogeneity, hindering expert specialization in vision MoE.

<div align="center">
  <img src="https://img.alicdn.com/imgextra/i1/O1CN01rduqOi22t7gZTZwFG_!!6000000007177-2-tps-1722-1292.png" width="70%">
</div>

To this end, we present ProMoE, an MoE framework featuring a two-step router with explicit routing guidance that promotes expert specialization. Specifically, this guidance encourages the router to partition image tokens into conditional and unconditional sets via conditional routing according to their functional roles, and refine the assignments of conditional image tokens through prototypical routing with learnable prototypes based on semantic content. Moreover, the similarity-based expert allocation in latent space enabled by prototypical routing offers a natural mechanism for incorporating explicit semantic guidance, and we validate that such guidance is crucial for vision MoE. Building on this, we propose a routing contrastive loss that explicitly enhances the prototypical routing process, promoting intra-expert coherence and inter-expert diversity. Extensive experiments on ImageNet benchmark demonstrate that ProMoE surpasses state-of-the-art methods under both Rectified Flow and DDPM training objectives.

## 🤗 Overview

This codebase supports:

* **Baselines:** Dense-DiT, TC-DiT, EC-DiT, DiffMoE, and their variants.
* **Proposed Models:** ProMoE variants (S, B, L, XL) with Token-Choice (TC) and Expert-Choice (EC) routing.
* **VAE Latent Preprocessing:** Pre-encode raw images into latents and cache them for faster training; supports multi-GPU parallel processing.
* **Sampling and Metric Evaluation:** Image sampling, Inception feature extraction, and calculation of FID, IS, sFID, Precision, and Recall; supports multi-GPU parallel processing.


## 🔥 Updates
- __[2026.02]__: Release the training, sampling, and evaluation code of ProMoE.
- __[2026.01]__: 🎉 Our paper has been accepted by **ICLR 2026**!
- __[2025.10]__: Release the [paper](https://arxiv.org/abs/2510.24711) of ProMoE.


## ⚙️ Preparation
### 1. Requirements & Installation
```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```

### 2. Dataset Preparation

Download [ImageNet](http://image-net.org/download) dataset, and modify `cfg.data_path` in `config.py`.

### 3. VAE Latent Preprocessing (Optional)

For faster training and more efficient GPU usage, you can **precompute VAE latents** and train with `cfg.use_pre_latents=True`.

Run latent preprocessing:
```bash
# bash
ImageNet_path=/path/to/ImageNet

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python preprocess/preprocess_vae.py --latent_save_root "$ImageNet_path/sd-vae-ft-mse_Latents_256img_npz"
```

## 🚀 Training

Training is launched via `train.py` with a YAML config:
```bash
python train.py --config configs/004_ProMoE_L.yaml
```

**Notes:**

- This repository currently supports Rectified Flow with Logit-Normal sampling (following [SD3](https://arxiv.org/pdf/2403.03206)). For the DDPM implementation, please refer to this [repository](https://github.com/KlingTeam/DiffMoE/tree/main).
- By default, ProMoE utilizes Token-Choice routing. However, for DDPM-based training, we recommend using Expert-Choice in `models/models_ProMoE_EC.py`.
- Configuration files for all baseline models are provided in the `configs` directory.
- All results reported in the paper are obtained with `qk_norm=False`. For extended training steps (>2M steps), we suggest enabling `qk_norm=True` to ensure training stability.


## 💫 Sampling

Image generation is performed via the `sample.py` script, utilizing the same YAML configuration file used for training.

```bash
# use default setting
python sample.py --config configs/004_ProMoE_L.yaml

# use custom setting
python sample.py \
  --config configs/004_ProMoE_L.yaml \
  --step_list_for_sample 200000,300000 \
  --guide_scale_list 1.0,1.5,4.0 \
  --num_fid_samples 10000
```

**Notes:**

- By default, the script loads the checkpoint at **500k steps** and generates **50,000 images**, sweeping across guidance scales (CFG) of **1.0** and **1.5**.
- Generated images are saved as PNG files in the `sample/` directory within the same parent directory as the checkpoint folder. Filenames include both the sample index and class label.
- If you only want to calculate FID, you can set `cfg.save_inception_features=True` to save Inception features and reduce `cfg.save_img_num`.


## 📝 Evaluation

We follow the standard evaluation protocol outlined in [openai's guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations). All relevant code is located in the `evaluation` directory.

### 1. Environment Setup
Since the evaluation pipeline relies on TensorFlow, we strongly recommend creating a dedicated environment to avoid dependency conflicts.
```bash
conda create -n promoe_eval python=3.9 -y
conda activate promoe_eval
cd evaluation
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### 2. Download Reference Batch
Download the reference statistics file [VIRTUAL_imagenet256_labeled.npz](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) (for 256x256 images) and place it in the `evaluation` directory.

### 3. Execution
To calculate the metrics, run the evaluation script by specifying the path to your folder of generated images.
```bash
python run_eval.py /path/to/generated/images
```

## Acknowledgements

This code is built on top of [DiffMoE](https://github.com/KlingTeam/DiffMoE), [DiT](https://github.com/facebookresearch/DiT), and [guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations). We thank the authors for their great work.


## 🌟 Citation

If you find this code useful for your research, please cite our paper:

```bibtex
@inproceedings{wei2026promoe,
  title={Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance},
  author={Wei, Yujie and Zhang, Shiwei and Yuan, Hangjie and Han, Yujin and Chen, Zhekai and Wang, Jiayu and Zou, Difan and Liu, Xihui and Zhang, Yingya and Liu, Yu and others},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```