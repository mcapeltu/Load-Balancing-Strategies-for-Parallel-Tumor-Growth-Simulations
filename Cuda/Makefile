ct: 
	clear
	nvcc --generate-line-info SOURCES/crecimientoTumoral.cu -o BIN/ct
	
ct2: 
	nvcc SOURCES/crecimientoTumoral2.cu -o BIN/ct
	
rm:
	rm -f TABLES/*	
	rm -f IMAGES/*	
	rm -f OUTPUT/*
	
run: ct rm
	./BIN/ct
	python3 SOURCES/grafiquear.py

run2: ct2 rm
	./BIN/ct
	python3 SOURCES/grafiquear.py
	
clean:
	rm -f BIN/*
