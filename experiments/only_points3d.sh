python train.py \
  gs.dataset.source_path=/root/data/tandt/truck \
  gs.dataset.model_path=/test/ \
  init_wC.matches_per_ref=10000 \
  init_wC.nns_per_ref=3 \
  init_wC.num_refs=180 \
  only_init_with_corr=True