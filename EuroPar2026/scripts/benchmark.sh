#!/bin/bash
mkdir -p RESULTS

echo "========================================="
echo " Benchmark CRM (10 ejecuciones)"
echo "========================================="

for i in $(seq 1 10); do
	    echo "CRM run $i"
	        ./BIN/ct_CRM | tee RESULTS/crm_$i.txt
	done

	echo
	echo "========================================="
	echo " Benchmark FRM (10 ejecuciones)"
	echo "========================================="

	for i in $(seq 1 10); do
		    echo "FRM run $i"
		        ./BIN/ct_FRM | tee RESULTS/frm_$i.txt
		done

		echo
		echo "========================================="
		echo " Benchmark finalizado"
		echo "========================================="
