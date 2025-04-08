1. Download cuda 11.3 toolkit (need gcc-9) (`wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run sh cuda_11.3.1_465.19.01_linux.run`)
2. use cuda-11.3 to download environment (cuda 11.3)
3. `pip install -r requirements.txt`
4. `pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"`
5. `git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git && cd Pointnet2_PyTorch` ; `cd pointnet2_ops_lib && pip install ./`
6. `pip install typing_extensions --upgrade`
7. (opt) `sudo apt install libgl1-mesa-glx`
8. `pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl`