python train.py \
  train.gs_epochs=30000 \
  train.no_densify=True \
  gs.dataset.source_path=/root/data/tandt/truck \
  gs.dataset.model_path=/root/output/truck_edgs_10k_matches_per_ref \
  init_wC.matches_per_ref=10000 \
  init_wC.nns_per_ref=3 \
  init_wC.num_refs=180