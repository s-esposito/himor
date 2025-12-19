conda create -n himor python=3.10 -y
conda activate himor
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
pip install -r requirements.txt
pip install "numpy<2"