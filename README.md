# HiMoR: Monocular Deformable Gaussian Reconstruction with Hierarchical Motion Representation (CVPR 2025)

### [Project Page](https://pfnet-research.github.io/himor/) | [arXiv](https://arxiv.org/abs/2504.06210)

![video](assets/mochi.gif)

## Installation
Please follow the instructions below to set up the environment:

### Quick Setup (Recommended)
```bash
bash setup.sh
```

### Manual Setup
```bash
# Create a new conda environment
conda create -n himor python=3.10 -y
conda activate himor

# Install CUDA toolkit
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

# Install PyTorch with CUDA 12.1 support
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install additional packages
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

# Install Python dependencies
pip install -r requirements.txt

# Install cuML via conda (IMPORTANT: must be done after pip requirements)
# cuML must be installed via conda to properly manage RAPIDS ecosystem dependencies
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.12 python=3.10 cuda-version=12.1 -y

# Reinstall scikit-learn (may have been removed by cuML installation)
conda install scikit-learn -y

# Install numpy and cupy
pip install "numpy<2"
pip install cupy-cuda12x
```

### Important Notes
- **cuML must be installed via conda**, not pip, to avoid CUDA compatibility issues
- The transformers library is pinned to version 4.47.1 for compatibility with PyTorch 2.2.0
- CUDA 12.1 is required for proper GPU acceleration

## Data preparation
### iPhone Dataset
Download the preprocessed iPhone dataset from [here](https://github.com/vye16/shape-of-motion?tab=readme-ov-file#evaluation-on-iphone-dataset) and place it under `./data/iPhone/`. Pretrained checkpoints are available [here](https://drive.google.com/file/d/1s8pTSbUrfhrADYdsdB1X2C-Hle-k0ZzB/view?usp=sharing).

### Nvidia Dataset
We use the dataset provided by [Gaussian Marbles](https://github.com/coltonstearns/dynamic-gaussian-marbles), with foreground masks recomputed using the preprocessing scripts from [Shape of Motion](https://github.com/vye16/shape-of-motion). Download the preprocessed dataset from [here](https://github.com/coltonstearns/dynamic-gaussian-marbles?tab=readme-ov-file#downloading-data) and place it under `./data/nvidia/`.

### Custom Dataset
To train on a custom dataset, please follow the instruction provided by [Shape of Motion](https://github.com/vye16/shape-of-motion) for preprocessing. Note that in our case, the data should be formatted following the iPhone dataset structure. 

## Visualization
To visualize results using an interactive viewer, first download the pretrained checkpoints, then run the following command:
```bash
python run_rendering.py --ckpt-path <path-to-ckpt>
```

## Training
### iPhone Dataset
For better reconstruction especially in background:
```bash
python run_training.py --work-dir ./outputs/paper-windmill --port 8888 data:iphone --data.data-dir ./data/iphone/paper-windmill --data.depth_type depth_anything_colmap --data.camera_type refined
```

In the paper, we report results using the original camera poses:
```bash
# First, align monocular depth with LiDAR depth.
python preproc/align_monodepth_with_lidar.py --lidar_depth_dir ./data/iphone/paper-windmill/depth/1x/ --input_monodepth_dir ./data/iphone/paper-windmill/flow3d_preprocessed/depth_anything/1x --output_monodepth_dir ./data/iphone/paper-windmill/flow3d_preprocessed/aligned_depth_anything_lidar/1x --matching_pattern "0*"

# Then, run training. 
python run_training.py --work-dir ./outputs/paper-windmill --port 8888 data:iphone --data.data-dir ./data/iphone/paper-windmill --data.depth_type depth_anything_lidar --data.camera_type original
```

### Nvidia Dataset
Train with the following command:
```bash
python run_training.py --work-dir ./outputs/Balloon1 --num_fg 20000 --num_bg 40000 --num_epochs 800 --port 8888 data:nvidia --data.data-dir ./data/nvidia/Balloon1 --data.depth_type lidar --data.camera_type original 
```

## Evaluation
Ensure that the checkpoint file `outputs/<dataset-name>/checkpoints/last.ckpt` is available. You can either obtain this by training the model or download the provided checkpoints.

### Render Images
Use the checkpoint to render images:
```bash
python run_evaluation.py --work-dir outputs/paper-windmill/ --ckpt-path outputs/paper-windmill/checkpoints/last.ckpt data:iphone --data.data-dir ./data/iphone/paper-windmill
```

### Compute Metrics
Evaluate the rendered images to compute quantitative metrics:
```bash
# For the iPhone dataset
PYTHONPATH="." python scripts/evaluate_iphone.py --data_dir ./data/iphone/paper-windmill --result_dir ./outputs/paper-windmill/ 

# For the Nvidia dataset
PYTHONPATH="." python scripts/evaluate_nvidia.py --data_dir ./data/nvidia/Balloon1/ --result_dir ./outputs/Balloon1/ 
```

## Citation
```
@inproceedings{liang2025himor,
    author    = {Liang, Yiming and Xu, Tianhan and Kikuchi, Yuta},
    title     = {{H}i{M}o{R}: Monocular Deformable Gaussian Reconstruction with Hierarchical Motion Representation},
    booktitle = {CVPR},
    year      = {2025},
}
```

## Acknowledgement
Our implementation builds on [Shape of Motion](https://github.com/vye16/shape-of-motion). We thank the authors for open-sourcing their code.
