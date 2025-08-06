#!/bin/bash

# Check if scene argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset/scene>"
    echo "Example: $0 mipnerf360/bicycle"
    exit 1
fi

scene=$1

method="gaussian-splatting"
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

/opt/conda/bin/conda run -n edgs python train.py -s /nvs-leaderboard-data/mipnerf360/bicycle/train -m /nvs-leaderboard-output/mipnerf360/bicycle/edgs

# Render the test split
# eg: python render.py --data /nvs-leaderboard-data/$scene/test --output /nvs-leaderboard-output/$scene/$method/ 

# At the end, move your renders into the `expected_output_folder`
# eg: mv /nvs-leaderboard-output/$scene/$method/train/ours_$iterations/renders $expected_output_folder
######## END OF YOUR CODE ########

# Record end time and show duration
end_time=$(date +%s)
echo $((end_time - start_time)) > /nvs-leaderboard-output/$scene/$method/training_time.txt
