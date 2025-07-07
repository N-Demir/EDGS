Exported `points3d.bin` do not include the EDGS initialized scale/rotation/opacity etc. It's just a pointcloud. Would be interesting to test down the road whether that makes a difference or not. Some simple heurestics for setting them could be helpful I'm sure? 

- scale seems to be set by distance of 3d point to the camera center
- opacity set by triangulation error quality (seems like a reasonable move)
---
Run with `modal run -d run.py --shell-file experiments/truck.sh`
---
I really like their usage of wandb and the outputs at the end of runs. Pretty. In the long run probably a good idea
---
Neat that the homogeneous camera matrices are used to
- select a representative subset of K cameras (first get K clusters with k-means, then select the camera closest to each cluster center)
- then to select nearest neighboring camera to each camera

Is this a valid approach generally? I should see it show up everywhere as a standard way to separate out a space of cameras
---
`init_gaussians_with_corr` seems to do the heavy lifting of getting the dense representation and then updating a `GaussianModel` with them. 

`GaussianModel` comes from the original 3dgs Inria repo 
---
Seems like a pretty simple technique of taking the pose estimates from colmap, finding neighboring images, running a "keypoint matching model" to get keypoints (all stuff I assume was already done by colmap?) and then triangulating the keypoint's positions based on some geometric math from the multiple keyframes.
Using these starting positions for 3dgs speeds everything up and improves the quality. Seems that something like vggt would give similar improvements if it was high enough quality...