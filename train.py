import os
from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
import omegaconf
import wandb
import hydra
from argparse import Namespace
from omegaconf import OmegaConf
import numpy as np
from read_write_model import Point3D, write_points3D_binary
import shutil


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
    
    print(f"Copying dataset from {source_dataset_path} to {new_dataset_path}...")
    
    # Copy the entire dataset folder
    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path)  # Remove if exists
    shutil.copytree(source_dataset_path, new_dataset_path)
    
    # Save points3D as binary file in the sparse/0 folder
    sparse_0_path = os.path.join(new_dataset_path, "sparse", "0")
    points3D_path = os.path.join(sparse_0_path, "points3D.bin")
    write_points3D_binary(points3D, points3D_path)
    print(f"Saved {len(points3D)} points to {points3D_path}")

    trainer.train(cfg.train)
    
    # All done
    wandb.finish()
    print("\nTraining complete.")

if __name__ == "__main__":
    main()

