#!/bin/bash

# Check if data_folder and output_folder arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <data_folder> <output_folder>"
    echo "Example: $0 /nvs-bench-data/mipnerf360/bicycle /nvs-bench-output/mipnerf360/bicycle/ours"
    exit 1
fi
data_folder=$1
output_folder=$2

######## START OF YOUR CODE ########
# 1) Train 
#   python train.py --data $data_folder --output $output_folder --eval
# 2) Render the test split
#   python render.py --data $data_folder/test --output $output_folder --eval
# 3) Move the renders into `$output_folder/test_renders`
#   mv $output_folder/test/ours_30000/renders $output_folder/test_renders

python train.py wandb.mode="disabled" \
    gs.dataset.source_path=$data_folder \
    gs.dataset.model_path=$output_folder \
    gs.dataset.eval=True \
    init_wC.use=True \
    init_wC.matches_per_ref=15_000 \
    init_wC.nns_per_ref=3 \
    init_wC.num_refs=180

python ./submodules/gaussian-splatting/render.py \
    --iteration 30000 \
    -s $data_folder \
    -m $output_folder \
    --eval

mv $output_folder/test/ours_30000/renders $output_folder/test_renders