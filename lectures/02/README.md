
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

template: titleslide

# Introduction to Performance Programming with Python
## 2

---

# Numba (reminder)

- Numba lets us select certain functions to be just-in-time (JIT) compiled

- When a Numba function is called for the first time, it is compiled into machine code using LLVM (for given argument types) and stored

- Subsequent calls with same argument types execute the machine code

```Python
import numpy as np
from numba import jit

@jit (nopython = True)
def array_sqrt(n, ain, aout):
    for i in range(n):
        aout[i] = np.sqrt(ain[i])

a_in  = np.array([4.0, 9.0, 16.0, 25.0], np.double)
a_out = np.zeros(4, np.double)

array_sqrt(a_in.size, a_in, a_out)
print(a_out)
```
`[2. 3. 4. 5.]`

---

# Numba overview

.center[![:scale_img 90%](numba_flowchart.png)]

[source](https://github.com/ContinuumIO/gtc2017-numba)

---

# Numba performance - `nopython` mode

 - A jit-decorated function should run much faster if all of its operations can be compiled into machine code, as there is then no involvement of the (much slower) Python interpreter

- This requires that:
  - all native types in the function can be inferred
  - no new Python objects are created in the function

- If both are true, the Numba-compiled function code does not try to use the interpreter's Python/C API (for type checking or Python object handling)

- `nopython=True` in the `@jit` decorator means the function code will only run if this is the case
 - A warning is issued if these criteria cannot be met, and Numba will fall back to a less optimised mode

---

# PyPy vs Numba

- Numba and PyPy both provide JIT optimising compilation functionality for Python code
 - Numba works *with* the standard Python interpreter, i.e. CPython
 - PyPy is a separate standalone Python compiler implementation

- PyPy usage:
 - `pypy myScript.py`

- PyPy can offer considerable speedup, but several issues hamper uptake:
  - only implements core Python functionality
  - does not support integration with all of NumPy
  - not compatible with the majority of packages in the Python ecosystem

- Numba also does not support usage of every single Numpy feature, but code will still run as normal

---

# Cython

- Extension of the Python language
 - Essentially Python + declaration of static C data types

- Cython compiler converts Cython code into C code

- Resulting C code is compiled (`gcc`) into a shared library available to import into Python

- Static typing and ahead-of-time (AOT) compilation provide improved performance similar to compiled C code

- Automatically generates Python wrappers and Python/C API calls 
 - Unlike `ctypes`-based interfacing of Python with C!
 - Code can manipulate both Python and C variables, conversions occur automatically wherever possible

---

# Cython - an example

Pure Python (`fib.py`)

```Python
# Returns the nth Fibonacci number
def fib(n):
    a = 0.0
    b = 1.0
    for i in range(n):
        a, b = a + b, a
    return a
```
Could compile using Cython, but haven't defined C types so may not be faster

Can use Cython syntax to declare static C types, save this as `fib.pyx`:

```Cython
# Returns the nth Fibonacci number
def fib(int n):
  cdef int i
  cdef double a = 0.0
  cdef double b = 1.0
  for i in range(n):
    a, b = a + b, a
  return a
```

---

# Cython - an example

To use, need to create `setup.py` file:
```Python
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fib.pyx"),
)
```
Then build the C extension module (`fib.c` and `fib.cpython.so`) with command:
```
$ python setup.py build_ext --inplace
```
To use fib.cpython.so in a Python script, just do:
```Python
import fib
fib.fib(200)

```

---

# Cython - an example

Fibonacci example gives the following performance results:

 Implementation                                | Runtime | Speedup
-----------------------------------------------|---------|---------
 Pure Python                                   | 9.21 µs |    1 `x`
 Cythonized Pure Python (no static C-types)    | 1.38 µs |    7 `x`
 Cythonized Cython                             |  245 ns |   37 `x`

- Note: C files are 2500 - 3000 lines long!

---

# Cython

Advantages:
 - Cython code can call C-based numerical libraries without passing through Python/C API bottleneck
   - e.g. fast access to Numpy arrays using standard Python/Numpy syntax

- Automatically performs runtime checks, e.g. out-of-bounds array access

- Can manually manage memory of C structures if desired through `malloc/free` (Python objects garbage collected as normal)

- Allows incremental development and focused optimisation of hotspots
  - Not useful if you want to call out to C code that already exists - use interfacing approaches for that!
  
---

# Cython

Limitations / things to consider:

- Little/no speedup for conventional (non-C-translatable) Python code and Python native data structures (lists, tuples, dicts)

- Like with Numba, for Cython code to be fast it should avoid calling Python API or accessing Python-native data structures
  - Cythonize process can report which parts of Cython code have interactions with Python API, to target optimisation

---

template: titleslide
# Parallel Computing with Python?

---

# Threading and the Global Interpreter Lock

- A Python interpreter process can spawn threads (`import threading`, etc.)
 - implemented as OS-managed threads (pthreads in Linux)

Global Interpreter Lock (GIL) is a core part of the CPython interpreter and prevents parallel execution of threads

GIL consist of a **mutex lock** on threads - only one thread can execute bytecode in the interpreter at any time
 
Mutex lock acquisition and release implementation means multiple CPU-bound threads (typical for numerical computing) waste vast amounts of time 'battling' to obtain the GIL (massive contention)
 - Details: https://www.dabeaz.com/python/UnderstandingGIL.pdf

GIL is released for calls to e.g. C code, e.g. many Numpy functions, so can be less of an issue for numerical computing
 - Still problematic because numerical codes may intermingle CPU-bound Python with external calls

---

# Why the GIL?

- Created to ensure thread safety for Python garbage collection (avoiding race condition on reference counter)

- Also useful historically to allow calling out to C libraries that are not guaranteed to themselves be thread safe

- Easier to get faster single-threaded programs (no necessity to acquire or release locks on all data structures separately)

- Was easier to implement than lock-free interpreter or one with finer-grained locks!


---

# Parallel Computing with Python

- Next lecture we will look at various approaches for getting around the GIL

- Will go back and look at how native Python, Numba, Cython, etc. enable this using threading, processes, OpenMP and MPI










