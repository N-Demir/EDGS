import os
from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
import omegaconf
import wandb
import hydra
from argparse import Namespace
from omegaconf import OmegaConf
import numpy as np
from read_write_model import Point3D, write_points3D_binary, write_points3D_text
import shutil
import subprocess


def convert_gaussians_to_points3d_and_copy_dataset(trainer, cfg):
    """
    Convert gaussians to points3D format and copy essential dataset folders.
    
    Args:
        trainer: EDGSTrainer instance containing the gaussians
        cfg: Configuration object with dataset paths
    
    Returns:
        tuple: (points3D dictionary, new_dataset_path)
    """
    # Convert gaussians to points3D and save as binary file
    print("Converting gaussians to points3D...")
    
    # Get positions and colors from gaussians
    xyz = trainer.gaussians._xyz.detach().cpu().numpy()  # Shape: (N, 3)
    features_dc = trainer.gaussians._features_dc.detach().cpu().numpy()  # Shape: (N, 1, 3)
    
    # Convert features_dc to RGB colors (assuming they're in the range [0, 1])
    # features_dc has shape (N, 1, 3), we need to squeeze and convert to uint8
    rgb = (features_dc.squeeze(1) * 255).astype(np.uint8)  # Shape: (N, 3)
    
    # Create points3D dictionary
    points3D = {}
    for i in range(len(xyz)):
        points3D[i] = Point3D(
            id=i,
            xyz=xyz[i],
            rgb=rgb[i],
            error=0.0,  # Default error value
            image_ids=np.array([]),  # Empty array since we don't have image associations
            point2D_idxs=np.array([])  # Empty array since we don't have 2D point associations
        )
    
    # Create new dataset folder with _edgs suffix
    source_dataset_path = cfg.gs.dataset.source_path
    dataset_name = os.path.basename(source_dataset_path)
    new_dataset_name = dataset_name + "_edgs"
    new_dataset_path = os.path.join(os.path.dirname(source_dataset_path), new_dataset_name)
    
    print(f"Copying essential dataset folders from {source_dataset_path} to {new_dataset_path}...")
    
    # Remove existing directory if it exists
    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path)
    
    # Create the new directory structure
    os.makedirs(new_dataset_path, exist_ok=True)
    
    # Copy only the images folder using rsync with progress
    source_images_path = os.path.join(source_dataset_path, "images")
    new_images_path = os.path.join(new_dataset_path, "images")
    if os.path.exists(source_images_path):
        print(f"Copying images folder using rsync...")
        # rsync -av --progress: archive mode, verbose, show progress
        cmd = ['rsync', '-av', '--progress', source_images_path + '/', new_images_path + '/']
        subprocess.run(cmd, check=True)
        print("Images folder copied successfully!")
    else:
        print(f"Warning: images folder not found at {source_images_path}")
    
    # Copy only the sparse/0 folder using rsync with progress
    source_sparse_0_path = os.path.join(source_dataset_path, "sparse", "0")
    new_sparse_path = os.path.join(new_dataset_path, "sparse")
    new_sparse_0_path = os.path.join(new_sparse_path, "0")
    
    if os.path.exists(source_sparse_0_path):
        print(f"Copying sparse/0 folder using rsync (excluding .ply files)...")
        os.makedirs(new_sparse_path, exist_ok=True)
        # rsync -av --progress --exclude='*.ply': archive mode, verbose, show progress, exclude .ply files
        cmd = ['rsync', '-av', '--progress', '--exclude=*.ply', source_sparse_0_path + '/', new_sparse_0_path + '/']
        subprocess.run(cmd, check=True)
        print("Sparse/0 folder copied successfully! (excluding .ply files)")
    else:
        print(f"Warning: sparse/0 folder not found at {source_sparse_0_path}")
        # Create empty sparse/0 directory if it doesn't exist
        os.makedirs(new_sparse_0_path, exist_ok=True)
    
    # Save points3D files in the sparse/0 folder, overwriting only existing ones
    # Check what point cloud files exist in the original sparse/0 folder
    original_has_bin = os.path.exists(os.path.join(source_sparse_0_path, "points3D.bin"))
    original_has_txt = os.path.exists(os.path.join(source_sparse_0_path, "points3D.txt"))
    
    # Overwrite binary file if it existed in the original
    if original_has_bin:
        points3D_bin_path = os.path.join(new_sparse_0_path, "points3D.bin")
        write_points3D_binary(points3D, points3D_bin_path)
        print(f"Saved {len(points3D)} points to {points3D_bin_path} (binary format)")
    
    # Overwrite text file if it existed in the original
    if original_has_txt:
        points3D_txt_path = os.path.join(new_sparse_0_path, "points3D.txt")
        write_points3D_text(points3D, points3D_txt_path)
        print(f"Saved {len(points3D)} points to {points3D_txt_path} (text format)")
    
    # If neither existed, default to binary format
    if not original_has_bin and not original_has_txt:
        print("It seems unlikely that the original sparse/0 folder had neither binary nor text point cloud files. but we'll write a binary file anyway.")
        points3D_bin_path = os.path.join(new_sparse_0_path, "points3D.bin")
        write_points3D_binary(points3D, points3D_bin_path)
        print(f"Saved {len(points3D)} points to {points3D_bin_path} (binary format, default)")
    
    return points3D, new_dataset_path


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    _ = wandb.init(entity=cfg.wandb.entity,
                   project=cfg.wandb.project,
                   config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                   tags=[cfg.wandb.tag], 
                   name = cfg.wandb.name,
                   mode = cfg.wandb.mode)
    omegaconf.OmegaConf.resolve(cfg)
    set_seed(cfg.seed)

    # Init output folder
    print("Output folder: {}".format(cfg.gs.dataset.model_path))
    os.makedirs(cfg.gs.dataset.model_path, exist_ok=True)
    with open(os.path.join(cfg.gs.dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        params = {
                "sh_degree": 3,
                "source_path": cfg.gs.dataset.source_path,
                "model_path": cfg.gs.dataset.model_path,
                "images": cfg.gs.dataset.images,
                "depths": "",
                "resolution": -1,
                "_white_background": cfg.gs.dataset.white_background,
                "train_test_exp": False,
                "data_device": cfg.gs.dataset.data_device,
                "eval": False,
                "convert_SHs_python": False,
                "compute_cov3D_python": False,
                "debug": False,
                "antialiasing": False   
                    }
        cfg_log_f.write(str(Namespace(**params)))

    # Init both agents
    gs = hydra.utils.instantiate(cfg.gs) 

    # Init trainer and launch training
    trainer = EDGSTrainer(GS=gs,
        training_config=cfg.gs.opt,
        device=cfg.device)
    
    trainer.load_checkpoints(cfg.load)
    trainer.timer.start()
    trainer.init_with_corr(cfg.init_wC)

    # Convert gaussians and copy dataset
    points3D, new_dataset_path = convert_gaussians_to_points3d_and_copy_dataset(trainer, cfg)

    trainer.train(cfg.train)
    
    # All done
    wandb.finish()
    print("\nTraining complete.")

if __name__ == "__main__":
    main()

