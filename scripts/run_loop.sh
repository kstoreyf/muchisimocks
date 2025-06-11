#!/bin/bash

n_train_arr=(500 1000 2000 4000 6000 8000 10000)
#n_train_arr=(4000 6000 8000)
#n_train_arr=(2000)
#n_train_arr=(10000)
tag_stats_arr=("_pk", "_bispec")
#tag_stats_arr=("_pk" "_bispec" "_pk_bispec")  # Add more stats if needed, e.g. ("pk" "pk_bispec")

for n_train in "${n_train_arr[@]}"; do
    for tag_stats in "${tag_stats_arr[@]}"; do
        config_file="../configs/configs_test/config_TRAIN_muchisimocks${tag_stats}_p5_n10000_biaszen_p4_n10000_ntrain${n_train}_TEST_muchisimocks${tag_stats}_quijote_p0_n1000_b1000_p0_n1_mean.yaml"
        echo "Launching: python run_inference.py --config-test=${config_file}"
        config_base=$(basename "$config_file" .yaml)
        job_name="logs/run_inf_${config_base}.out"
        echo "Job name: ${job_name}"
        python run_inference.py --config-test="${config_file}" > $job_name 2>&1 &
    done
done

wait  # Wait for all background jobs to finish
echo "All inference jobs finished."