for seed in 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010
do
    out="../DataMod2026/results/multiseed/seed_${seed}"

    mkdir -p "$out"

    echo "====================================="
    echo "Running seed ${seed}"
    echo "====================================="

    ./BIN/ct_datamod_seeds \
        --seed "$seed" \
        --output "$out"
done
