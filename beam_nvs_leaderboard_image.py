###### How to edit this file ######
# Docker and Dockerfiles are quite simple:
# - a dockerfile is the set of instructions for getting a fresh machine ready to run your code
# - start by defining a base image (FROM ...) based on the cuda and torch version you want. This gets the hard gpu driver stuff out of the way
# - set env vars with ENV ..., change directories with WORKDIR ..., and run commands with RUN ...
# - avoid using conda installs (just replace them with pip installs) because getting conda initialized in docker is a pain
# 
# Beam will handle building the docker image from this file, but you can also build it yourself and run it wherever you want
from beam import Image

image = (
    Image(
        base_image="pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel",
    )
    .with_envs(
        {
            "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.9;9.0",
            "DEBIAN_FRONTEND": "noninteractive",
            "TZ": "America/New_York",
        }
    )
    .add_commands(["apt-get update && apt-get install -y openssh-server git wget unzip cmake build-essential ninja-build libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev && rm -rf /var/lib/apt/lists/*",
        "git clone -b nvs-leaderboard https://github.com/N-Demir/EDGS.git --recursive .",
        "pip install -e submodules/gaussian-splatting/submodules/diff-gaussian-rasterization",
        "pip install -e submodules/gaussian-splatting/submodules/simple-knn",
        "pip install -e submodules/RoMa",
        "pip install pycolmap",
        "pip install wandb hydra-core tqdm torchmetrics lpips matplotlib rich plyfile imageio imageio-ffmpeg numpy==1.26.4",
        "pip install gradio plotly scikit-learn moviepy==2.1.1 ffmpeg open3d"
    ])
)