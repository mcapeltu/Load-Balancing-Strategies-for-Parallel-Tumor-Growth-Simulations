ct: 
	clear
	nvcc --generate-line-info SOURCES/crecimientoTumoral.cu -o BIN/ct
	
ct2: 
	nvcc --generate-line-info SOURCES/crecimientoTumoral2.cu -o BIN/ct

ctDLB: 
	clear
	nvcc --generate-line-info SOURCES/crecimientoTumoralDLB.cu -o BIN/ct
	
rm:
	rm -f TABLES/*	
	rm -f IMAGES/*	
	rm -f OUTPUT/*
	rm -f TIMES/*
	
run: ct rm
	./BIN/ct
	python3 SOURCES/grafiquear.py

run2: ct2 rm 
	./BIN/ct
	python3 SOURCES/grafiquear.py
	
runDLB: ctDLB rm
	./BIN/ct
	python3 SOURCES/grafiquear.py
	
clean: rm
	rm -f BIN/*
