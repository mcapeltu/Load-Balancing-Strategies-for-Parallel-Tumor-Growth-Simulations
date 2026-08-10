#!/usr/bin/env bash

set -euo pipefail

RUNS=10
MAX_START_TEMP=60
POLL_SECONDS=5
STAMP=$(date +"%Y%m%d_%H%M%S")
OUTDIR="RESULTS_THERMAL_${STAMP}"

mkdir -p "$OUTDIR"

wait_for_cooling() {
    local temperature

    while true; do
        temperature=$(nvidia-smi \
            --query-gpu=temperature.gpu \
            --format=csv,noheader,nounits | head -n 1 | tr -d ' ')

        if ! [[ "$temperature" =~ ^[0-9]+$ ]]; then
            echo "Could not read GPU temperature."
            exit 1
        fi

        echo "GPU temperature: ${temperature} °C"

        if (( temperature <= MAX_START_TEMP )); then
            break
        fi

        echo "Waiting for GPU to cool to ${MAX_START_TEMP} °C or below..."
        sleep "$POLL_SECONDS"
    done
}

run_test() {
    local name="$1"
    local executable="$2"
    local run_number="$3"

    local prefix="${OUTDIR}/${name}_${run_number}"
    local telemetry="${prefix}_telemetry.csv"
    local output="${prefix}.txt"

    wait_for_cooling

    echo
    echo "Starting ${name}, run ${run_number}"
    echo "$(date --iso-8601=seconds)" > "${prefix}_start.txt"

    nvidia-smi \
        --query-gpu=timestamp,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,power.draw \
        --format=csv \
        -l 1 > "$telemetry" &

    local monitor_pid=$!

    set +e
    "$executable" | tee "$output"
    local program_status=${PIPESTATUS[0]}
    set -e

    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true

    echo "$(date --iso-8601=seconds)" > "${prefix}_end.txt"

    if (( program_status != 0 )); then
        echo "${name} run ${run_number} failed."
        exit "$program_status"
    fi

    grep "Tiempo:" "$output" | awk '{print $NF}' \
        >> "${OUTDIR}/${name}_times.txt"
}

echo "Results directory: $OUTDIR"
echo "Maximum starting temperature: ${MAX_START_TEMP} °C"

mkdir -p TABLES TIMES OUTPUT IMAGES

echo
echo "Performing unrecorded warm-up..."
wait_for_cooling
./BIN/ct_CRM > /dev/null

for i in $(seq 1 "$RUNS"); do
    # Alternate the order to avoid systematically favouring one version.
    if (( i % 2 == 1 )); then
        run_test "CRM" "./BIN/ct_CRM" "$i"
        run_test "FRM" "./BIN/ct_FRM" "$i"
    else
        run_test "FRM" "./BIN/ct_FRM" "$i"
        run_test "CRM" "./BIN/ct_CRM" "$i"
    fi
done

echo
echo "Benchmark completed."
echo "Results stored in: $OUTDIR"  
