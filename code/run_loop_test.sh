REPO=/home/kstoreyf/muchisimocks
CFGDIR=$REPO/configs/configs_test
#k_tags=("" "_kb0.1" "_kb0.15" "_kb0.2" "_kb0.25" "_kb0.3" "_kb0.35")
k_tags=("" "_kpgm0.1" "_kpgm0.15" "_kpgm0.2" "_kpgm0.25" "_kpgm0.3" "_kpgm0.35")
nbars=(0.00011 0.00054)
tag_stats=("pk_pgm")
for k_tag in "${k_tags[@]}"; do
  for nb in "${nbars[@]}"; do
    if [ -z "$k_tag" ]; then
      mid_train="${tag_stats}_p5_n10000_biasnoisenest_p9_n320000_noise_unit_p5_n10000_rp_bx32_ntrain10000_best-rand30"
      test_suffix="TEST_shame_${tag_stats}_nbar${nb}.yaml"
    else
      k_tag="${k_tag#_}"
      mid_train="${tag_stats}_${k_tag}_p5_n10000_biasnoisenest_p9_n320000_noise_unit_p5_n10000_rp_bx32_ntrain10000_best-rand30"
      test_suffix="TEST_shame_${tag_stats}_${k_tag}_nbar${nb}.yaml"
    fi
    python "$REPO/code/run_inference.py" \
      --config-test="$CFGDIR/config_TRAIN_muchisimocks_${mid_train}_${test_suffix}"
  done
done