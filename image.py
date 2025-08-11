from beam import Image

image = (
    Image(base_image="nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04")
    .with_envs(
        {
            "DEBIAN_FRONTEND": "noninteractive",
            "TZ": "America/New_York",
            "TORCH_CUDA_ARCH_LIST": "7.5;8.0;8.9;9.0",
        }
    )
    .add_commands(
        [
            "apt-get update -y",
            "apt-get install -y openssh-server git wget unzip cmake build-essential ninja-build libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev",
            "rm -rf /var/lib/apt/lists/*",  
        ]
    )
    .add_commands([
        "pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121",
    ])
    # .add_commands(
    #     [
    #         "git clone -b nvs-leaderboard https://github.com/N-Demir/EDGS.git --recursive .",
    #         "pip install -e submodules/gaussian-splatting/submodules/diff-gaussian-rasterization",
    #         "pip install -e submodules/gaussian-splatting/submodules/simple-knn",
    #         "pip install -e submodules/RoMa",
    #         "pip install pycolmap",
    #         "pip install wandb hydra-core tqdm torchmetrics lpips matplotlib rich plyfile imageio imageio-ffmpeg numpy==1.26.4",
    #         "pip install gradio plotly scikit-learn moviepy==2.1.1 ffmpeg open3d",
    #     ]
    # )
)