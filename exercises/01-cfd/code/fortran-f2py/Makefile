#
# Build shared library to allow Fortran Jacobi function to be
# called from Python. Uses f2py and requires a Fortran compiler
# to be installed.
#
# Note that it is also possible to use the 'single-stage'
# approach
#   f2py -c -m jacobi jacobi.f90
#
# which bypasses the 'signature' file.
#
#
# ---- Compile and Link --------------------------- 
jacobi.so: jacobi.f90 jacobi.pyf
	f2py -c jacobi.pyf jacobi.f90

# ---- Create Signature ------------------
jacobi.pyf: jacobi.f90
	f2py jacobi.f90 -m jacobi -h jacobi.pyf

clean:
	rm -rf *.so *.pyf  __pycache__ *pyc
