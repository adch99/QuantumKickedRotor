compile:
	python setup.py build_ext --inplace

clean:
	rm -rf */*.pyc */*.so */*.c
	rm -rf build/*

run:
	python bin/run_entanglement.py > logs/output.log 2> logs/errors.log

all:
	compile
