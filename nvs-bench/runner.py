# You shouldn't need to edit this file, but feel free to take a look at how things are called and run remotely
import os
import shutil
import subprocess
import time
from pathlib import Path
from contextlib import contextmanager
import shutil
import modal
from .image import image, modal_volumes, data_volume, output_volume, method_name

app = modal.App(
    "nvs-bench",
    image=(
        image  # If using Dockerfile, replace with `modal.Image.from_dockerfile("Dockerfile")`
        # Overwrite build repo (which is only pulled in once for install) with the current local working directory
        .add_local_dir(Path.cwd(), f"/root/{method_name}")
    ),
    volumes=modal_volumes,
)


@contextmanager
def log_max_gpu_memory(log_file: str):
    """Context manager to track GPU memory usage and log maximum memory to a file."""
    import gpu_tracker as gput

    with gput.Tracker(sleep_time=0.1, gpu_ram_unit="megabytes", disable_logs=True) as t:
        yield

    with open(log_file, "w") as f:
        f.write(str(int(t.resource_usage.max_gpu_ram.system))) # type: ignore

@contextmanager
def log_time(log_file: str):
    """Context manager to track execution time and log it to a file."""
    start_time = time.time()
    yield
    duration = time.time() - start_time

    with open(log_file, "w") as f:
        f.write(f"{duration:.2f}\n")


@app.function(
    timeout=3600 * 8,
    gpu="L40S",
)
def eval(data: str):
    data_volume.reload()

    data_folder = Path(f"/nvs-bench-data/{data}/")
    output_folder = Path(f"/nvs-bench-output/{data}/{method_name}/")

    # Check if the data exists
    if not data_folder.exists():
        # Download from gcs
        dataset = data.split("/")[0]
        os.system(f"mkdir -p /nvs-bench-data/{dataset}/")
        os.system(f"gsutil -m cp -r gs://nvs-bench/data/{data} /nvs-bench-data/{dataset}/")

    # Clean output folder
    shutil.rmtree(output_folder, ignore_errors=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    with log_max_gpu_memory(f"{output_folder}/max_gpu_memory.txt"):
        with log_time(f"{output_folder}/time.txt"):
            os.system(f"bash nvs-bench/eval.sh {data_folder} {output_folder}")

    output_volume.commit()


def full_eval():
    """Runs without waiting for each scene to finish"""
    BENCHMARK_SCENES = [
        # Mipnerf360
        "mipnerf360/bicycle",
        "mipnerf360/treehill",
        "mipnerf360/stump",
        "mipnerf360/room",
        "mipnerf360/kitchen",
        "mipnerf360/garden",
        "mipnerf360/flowers",
        "mipnerf360/counter",
        "mipnerf360/bonsai",
        # Tanks and Temples
        "tanksandtemples/truck",
        "tanksandtemples/train",
        # DeepBlending
        "deepblending/playroom",
        "deepblending/drjohnson",
        # ZipNerf
        "zipnerf/alameda",
        "zipnerf/berlin",
        "zipnerf/london",
        "zipnerf/nyc",
    ]

    for scene in BENCHMARK_SCENES:
        eval.spawn(scene)


@app.local_entrypoint()
def main(data: str | None = None):
    """Run train/render on a scene (eg: mipnerf360/bicycle) or if not provided the full eval"""
    if data is not None:
        # Assert there's only one / in the data
        assert data.count("/") == 1, "data must be in the format <dataset>/<scene>"
        eval.remote(data)
    else:
        full_eval()
