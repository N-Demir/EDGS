import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import modal

app = modal.App("gsplat", image=modal.Image.from_dockerfile(Path(__file__).parent / "Dockerfile")
    # GCloud
    .add_local_file(Path.home() / "gcs-tour-project-service-account-key.json", "/root/gcs-tour-project-service-account-key.json", copy=True)
    .run_commands(
        "gcloud auth activate-service-account --key-file=/root/gcs-tour-project-service-account-key.json",
        "gcloud config set project tour-project-442218",
        "gcloud storage ls"
    )
    .env({"GOOGLE_APPLICATION_CREDENTIALS": "/root/gcs-tour-project-service-account-key.json"})
    .run_commands("gcloud storage ls")
    # SSH server
    .apt_install("openssh-server")
    .run_commands(
        "mkdir -p /run/sshd" #, "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config", "echo 'root: ' | chpasswd" #TODO: uncomment this if the key approach doesn't work
    )
    .add_local_file(Path.home() / ".ssh/id_rsa.pub", "/root/.ssh/authorized_keys", copy=True)
    # Add Conda (for some reason necessary for ssh-based code running)
    .run_commands("conda init bash && echo 'conda activate base' >> ~/.bashrc")
    # Fix Git
    .run_commands("git config --global pull.rebase true")
    .run_commands("git config --global user.name 'Nikita Demir'")
    .run_commands("git config --global user.email 'nikitde1@gmail.com'")
    # Set CUDA Architecture (depends on the GPU)
    .env({"TORCH_CUDA_ARCH_LIST": "7.5"})
    # Add Our Code and Install EDGS
    .workdir("/root/workspace")
    # Clone EDGS repository
    .run_commands("git clone https://github.com/N-Demir/EDGS.git --recursive")
    .workdir("/root/workspace/EDGS")
    .run_commands("git submodule update --init --recursive")
    # Install submodules
    .run_commands("pip install -e submodules/gaussian-splatting/submodules/diff-gaussian-rasterization")
    .run_commands("pip install -e submodules/gaussian-splatting/submodules/simple-knn")
    # Install PyTorch and CUDA
    # .run_commands("conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y")
    # .run_commands("conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y")
    # Install COLMAP
    .run_commands("pip install pycolmap")
    # Install other dependencies
    .run_commands("pip install wandb hydra-core tqdm torchmetrics lpips matplotlib rich plyfile imageio imageio-ffmpeg")
    .run_commands("conda install numpy=1.26.4 -y -c conda-forge --override-channels")
    # Install RoMa
    .run_commands("pip install -e submodules/RoMa")
    .run_commands("conda install anaconda::jupyter --yes")
    # Install visualization dependencies
    .run_commands("pip install gradio")
    .run_commands("pip install plotly scikit-learn moviepy==2.1.1 ffmpeg")
    .run_commands("pip install open3d")
    # Get the latest code
    .run_commands("git pull", force_build=True)
)


LOCAL_PORT = 9090


def wait_for_port(host, port, q):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 22), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError("Waited too long for port 22 to accept connections") from exc
        q.put((host, port))


@app.function(
    timeout=3600 * 24,
    gpu="T4",
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("github-token")],
    volumes={
             "/root/data": modal.Volume.from_name("data", create_if_missing=True),
             "/root/output": modal.Volume.from_name("output", create_if_missing=True),
             "/root/ever_training": modal.Volume.from_name("ever-training", create_if_missing=True)}
)
def launch_ssh_2(q):
    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(target=wait_for_port, args=(host, port, q)).start()

        # Added these commands to get the env variables that docker loads in through ENV to show up in my ssh
        subprocess.run("env | awk '{print \"export \" $1}' > ~/env_variables.sh", shell=True)
        subprocess.run("echo 'source ~/env_variables.sh' >> ~/.bashrc", shell=True)

        subprocess.run(["/usr/sbin/sshd", "-D"])  # TODO: I don't know why I need to start this here

@app.local_entrypoint()
def main():
    import sshtunnel

    with modal.Queue.ephemeral() as q:
        launch_ssh_2.spawn(q)
        host, port = q.get()
        print(f"SSH server running at {host}:{port}")

        server = sshtunnel.SSHTunnelForwarder(
            (host, port),
            ssh_username="root",
            ssh_password=" ",
            remote_bind_address=("127.0.0.1", 22),
            local_bind_address=("127.0.0.1", LOCAL_PORT),
            allow_agent=False,
        )

        try:
            server.start()
            print(f"SSH tunnel forwarded to localhost:{server.local_bind_port}")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down SSH tunnel...")
        finally:
            server.stop()