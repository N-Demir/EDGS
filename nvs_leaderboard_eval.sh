#!/bin/bash

# Check if scene argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset/scene>"
    echo "Example: $0 mipnerf360/bicycle"
    exit 1
fi

scene=$1

method="edgs"
expected_output_folder="/nvs-leaderboard-output/$scene/$method/renders_test"

# Remove the output folder if it already exists
rm -rf /nvs-leaderboard-output/$scene/$method

# Record start time
start_time=$(date +%s)

######## START OF YOUR CODE ########
# TODO: Add an example
# Train using the train split in the dataset folder
# eg: python train.py --data /nvs-leaderboard-data/$scene/train --output /nvs-leaderboard-output/$scene/$method/

# If your dockerfile uses conda and you're facing a `run conda init before conda activate` error, I would just recommend
# being explicit about what python to use instead of trying to get the conda environment to work. Like so:

python train.py wandb.mode="disabled" \
    gs.dataset.source_path=/nvs-leaderboard-data/$scene/train \
    gs.dataset.model_path=/nvs-leaderboard-output/$scene/$method \
    train.gs_epochs=10 \
    gs.opt.save_iterations=[10] \
    init_wC.use=True \
    init_wC.matches_per_ref=15_000 \
    init_wC.nns_per_ref=3 \
    init_wC.num_refs=180

# Render the test split
# eg: python render.py --data /nvs-leaderboard-data/$scene/test --output /nvs-leaderboard-output/$scene/$method/ 
# for testing: python ./submodules/gaussian-splatting/render.py --iteration 10 -s /nvs-leaderboard-data/mipnerf360/bicycle/train -m /nvs-leaderboard-output/mipnerf360/bicycle/edgs --eval --skip_train
python ./submodules/gaussian-splatting/render.py \
    --iteration 10 \
    -s /nvs-leaderboard-data/$scene/train \
    -m /nvs-leaderboard-output/$scene/$method \
    --eval \
    --skip_train

# At the end, move your renders into the `expected_output_folder`
# eg: mv /nvs-leaderboard-output/$scene/$method/train/ours_$iterations/renders $expected_output_folder
######## END OF YOUR CODE ########

# Record end time and show duration
end_time=$(date +%s)
echo $((end_time - start_time)) > /nvs-leaderboard-output/$scene/$method/training_time.txt
