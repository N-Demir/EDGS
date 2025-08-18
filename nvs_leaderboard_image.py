###### How to edit this file ######
# Docker and Dockerfiles are quite simple:
# - a dockerfile is the set of instructions for getting a fresh machine ready to run your code
# - start by defining a base image (FROM ...) based on the cuda and torch version you want. This gets the hard gpu driver stuff out of the way
# - set env vars with ENV ..., change directories with WORKDIR ..., and run commands with RUN ...
# - avoid using conda installs (just replace them with pip installs) because getting conda initialized in docker is a pain
# 
# Beam will handle building the docker image from this file, but you can also build it yourself and run it wherever you want
from pathlib import Path

from modal import Image

image = (
    Image
    .from_registry("pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel")
    .env(
        {
            # Set Torch CUDA Compatbility to be for RTX 4090, T4, L40s, and A100
            # If using a different GPU, make sure its torch cuda architecture version is added to the list
            "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.9;9.0",
            # Set environment variable to avoid interactive prompts from installing packages
            "DEBIAN_FRONTEND": "noninteractive",
            "TZ": "America/New_York",
        }
    )
    # Install git and various other helper dependencies
    .run_commands(
        "apt-get update && apt-get install -y \
            openssh-server \
            git \
            wget \
            unzip \
            cmake \
            build-essential \
            ninja-build \
            libglew-dev \
            libassimp-dev \
            libboost-all-dev \
            libgtk-3-dev \
            libopencv-dev \
            libglfw3-dev \
            libavdevice-dev \
            libavcodec-dev \
            libeigen3-dev \
            libtbb-dev \
            libopenexr-dev \
            libxi-dev \
            libxrandr-dev \
            libxxf86vm-dev \
            libxxf86dga-dev \
            libxxf86vm-dev \
            && rm -rf /var/lib/apt/lists/*"
    )
    .workdir(f"/root/{Path.cwd().name}")

    ###### Your Code Here ######
    # Probably easiest to pull the repo from github, but you can also copy files from your local machine with COPY 
    # eg: RUN git clone https://github.com/graphdeco-inria/gaussian-splatting.git . --recursive
    .run_commands("git clone -b nvs-leaderboard https://github.com/N-Demir/EDGS.git --recursive .")
    # Install (avoid conda installs because they don't work well in dockerfile situations)
    # Separating these on separate lines helps if there are errors (previous lines will be cached) especially on the large package installs
    # eg:
    # RUN pip install submodules/diff-gaussian-rasterization
    # RUN pip install -e .
    # Note: If your run_commands step needs access to a gpu it's actually possible to do that through "run_commands(gpu='L40S', ...)"
    .run_commands("pip install -e submodules/gaussian-splatting/submodules/diff-gaussian-rasterization")
    .run_commands("pip install -e submodules/gaussian-splatting/submodules/simple-knn")
    .run_commands("pip install -e submodules/RoMa")
    # For COLMAP and pycolmap
    # Optionally install original colmap but probably pycolmap suffices
    # conda install conda-forge/label/colmap_dev::colmap
    # RUN pip install pycolmap
    .run_commands("pip install wandb hydra-core tqdm torchmetrics lpips matplotlib rich plyfile imageio imageio-ffmpeg numpy==1.26.4")
    # Stuff necessary for gradio and visualizations
    .run_commands("pip install gradio plotly scikit-learn moviepy==2.1.1 ffmpeg")
    .run_commands("pip install open3d")
)