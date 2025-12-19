#!/bin/bash
set -e  # Exit on error

echo "Creating conda environment..."
conda create -n himor python=3.10 -y
conda activate himor

echo "Installing CUDA toolkit..."
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

echo "Installing gsplat..."
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation

echo "Installing pytorch3d..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation

echo "Installing fused-ssim..."
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation

echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing cuML via conda (must be done after pip requirements)..."
# cuML must be installed via conda to properly manage RAPIDS ecosystem dependencies
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.12 python=3.10 cuda-version=12.1 -y

echo "Installing scikit-learn (may have been removed by cuML installation)..."
conda install scikit-learn -y

echo "Installing numpy and cupy..."
pip install "numpy<2"
pip install cupy-cuda12x

echo ""
echo "Installation complete!"
echo "To activate the environment, run: conda activate himor"