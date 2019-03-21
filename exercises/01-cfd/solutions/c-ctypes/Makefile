#
# Build shared library to allow Jacobi function to be
# called from Python.
#

jacobi.so: jacobi.c
	gcc -c -fPIC -Wall jacobi.c
	gcc -shared -o jacobi.so jacobi.o

test:	jacobi.so
	./cfd.py 1 1000
	diff velocity.dat ../verify/cfd_velocity_1_1000.dat

clean:
	rm -rf *.so *.o  __pycache__
