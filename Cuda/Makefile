ct: 
	clear
	nvcc --generate-line-info SOURCES/crecimientoTumoral.cu -o BIN/ct
	
ct2: 
	nvcc --generate-line-info SOURCES/crecimientoTumoralTiempos.cu -o BIN/ct

ctDLB: 
	clear
	nvcc --generate-line-info SOURCES/crecimientoTumoralDLB.cu -o BIN/ct
	
rm:
	rm -f TABLES/*	
	rm -f IMAGES/*	
	rm -f OUTPUT/*
	rm -f TIMES/*

exe:
	./BIN/ct
	#python3 SOURCES/grafiquear.py	
	
run: ct rm exe

runTiempos: ct2 rm exe
	
runDLB: ctDLB rm exe
	
clean: rm
	rm -f BIN/*
