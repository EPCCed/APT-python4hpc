
template: titleslide

# CFD Practical Summary

---

# CFD Practical Summary

Performance results on Cirrus for ./cfd 10 1000: 

Implementation             |   Runtime  | Speedup
---------------------------|------------|---------
ndarrays (loops)           |    103s    |   1 `x`
lists (loops)              |    46s     |   2 `x`
ndarrays (slicing) & numba |    2.23s   |  46 `x`
scipy convolve             |    1.46s   |  70 `x`
C -O0 (ctypes)             |    1.24s   |  80 `x`
ndarrays (slicing)         |    0.72s   |  140 `x`
Fortran -O3 (f2py)         |    0.26s   |  400 `x`
C -O3 (ctypes)             |    0.19s   |  540 `x`

- Runs performed using:
 - Python 3.6.4
 - Numpy 1.14.0
 - SciPy 1.0.0
 - gcc 8.2.0

---

# CFD Practical Conclusions

- Avoid explicit `for` loops
 - use slicing-based Numpy array indexing instead

- Numba doesn't always help, and places some limitations on supported Numpy features

- Interfacing with compiled and optimised C/Fortran code can yield significant speedups

---


# A note on .pyc files and performance

- `.pyc` files = bytecode saved to disk, ready to be executed by the Python VM

- Interpreter automatically creates these for `.py` files imported by a script

- If `file.pyc` is newer than `file.py`, Python VM executes the `pyc` directly
 - saves time performing an unnecessary translation step

- Can bias multiple-run performance tests unless removed inbetween runs
 - located in same dir as imported `py` file, or in `__pycache`__ subdir

---
