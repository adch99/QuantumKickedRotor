compile:
	python setup.py build_ext --inplace

clean:
	rm -rf */*.pyc */*.so */*.c
	rm -rf build/*

run:
	compile
	python bin/run_entanglement.py

all:
	compile
