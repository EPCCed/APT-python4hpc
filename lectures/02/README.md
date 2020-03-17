template: titleslide

# Introduction to Performance Programming with Python
## 2

---

# ...where were we?

Looked at how Python code is executed by the default interpreter and what that tells us about performance. For fast numerical computing, want to:
- Minimise overheads from dynamic typing & stack-based execution
- Exploit vectorisation

How?
- Use NumPy arrays (statically typed, fixed size)
  - mind efficient syntax to avoid creating temporaries
- Call precompiled functions
  - For elementwise operations on NumPy arrays *don't write explicit for loops*, use overloaded operators (`+`, `-`, `*`, `/`, `**`) and related [`ufuncs`](https://numpy.org/devdocs/reference/ufuncs.html) ("universal functions"), many compiled from C
  - Interface Python with compiled Fortran/C/C++ code by importing functions from shared library (many options, looked at `ctypes`)


---

# What's next?

Going to cover 4 other strategies for speeding up Python performance:

- **CPython extension modules**: extend CPython interpreter with C code

- **Cython**: make existing Python code fast...by making it more like C

- **PyPy**: alternative Python interpreter (`pypy` instead of CPython's `python`)
  - speed from just-in-time (JIT) compilation  

- **Numba**: 
  - JIT compilation using LLVM

---

template:titleslide
# CPython extension modules

---

# CPython extension modules

- CPython provides a C API (`#include "Python.h"`), allows you to:
  - Write C code that directly integrates with rest of CPython
  - Operate on built\-in objects (possibly avoid Python stack execution)
  - Rebuild interpreter with extension module to make importable

- Typical usage:
 - accelerator module: self-contained C code, for speed
 - wrapper module: interface with other C code (libraries)
 - low-level system access: interface with OS, hardware, ...
  
- NumPy extension modules:
 - accelerator modules: e.g. inner loops moved to C
 - wrapper modules: e.g. call Fortran/C linear algebra libraries
 - low-level: e.g. vectorise operations, control memory layout of arrays

---

# CPython extension modules

- Advantage: extremely flexible

- Disadvantages:
    - Requires thorough understanding of CPython and its Python/C API
        - Significant initial development effort
        - Difficult to maintain

    - Requires rebuild of interpreter
        - Barrier to usage and cross-platform portability
        - Not feasible in some situations

- Cython (next) partly automates creation of CPython extension modules
    - Does not require intimate knowledge of CPython
    - Eliminates need for interpreter rebuild

---

template:titleslide
# Cython

---

# Cython

- Enrich performance-critical Python code with additional syntax
 - Cython = Python + static declarations of C types + calls to C functions

- Cython compiler converts Cython code to C code that calls CPython's C API
    - Effectively creates a CPython extension module for you
    - Automatically generates Python wrappers and Python/C API calls 
        - Unlike manual Python-C interfacing using e.g. `ctypes`!
        - Cython code can manipulate both Python and C variables, conversions occur automatically wherever possible

- Resulting C code is compiled into a shared library for import in Python
    - Using conventional C compiler, e.g. `gcc`
    - Benefits from all the usual static typing-based ahead-of-time (AOT) compilation optimisations

---

# Cython example

Pure Python (**`fib.py`**)

```Python
# Returns the nth Fibonacci number
def fib(n):
    a = 0.0
    b = 1.0
    for i in range(n):
        a, b = a + b, a
    return a
```
Could already Cython-compile, but haven't defined C types so may not be faster. Can use Cython syntax to declare static C types (save as **`fib.pyx`**):

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

# Cython example

To use, need to create `setup.py` file:
```Python
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fib.pyx"),
)
```
Build C extension module (`fib.c` and `fib.cpython.so`) from the shell:
```
> python setup.py build_ext --inplace
```
To use `fib` function from `fib.cpython.so` in a Python script, just do:
```Python
import fib
fib.fib(200)

```

---

# Cython example

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
- Automates much of CPython extension module creation

- Allows incremental development and focused optimisation of hotspots

- Automatically performs runtime checks, e.g. out-of-bounds array access

- Allows creation of wrapper modules to interface Python with C-based high-performance numerical libraries 
    - similar approach as example:
        - `cdef` and type-define function
        - link appropriately during build of shared library

- Can also be used to call into Python code from C
  
---

# Cython

Limitations / things to consider:

- Little/no speedup for conventional (non-C-translatable) Python code and Python native data structures (lists, tuples, dicts)

- For Cython code to be fast it should avoid frequent calls to Python functions or access of pure-Python data structures
  - Cythonize process can report which parts of Cython code do this to target optimisation

- Explicit type annotation cumbersome

- Often requires restructuring code

- Code build becomes more complicated, more difficult to maintain


---

template:titleslide
# PyPy

---

# PyPy

- Alternative Python interpreter (i.e. not CPython)
    - bytecode compiler
    - bytecode evaluator interprets & executes bytecode (similar to CPython's virtual machine)

- Uses just-in-time (JIT) compilation
    - Code executation starts straightaway (like CPython)
    - During execution PyPy traces which operations within the bytecode evaluator most contribute to execution time (hotspot detection)
    - Attempts various optimisations on traced operations (even before generating machine code)
        - Constant folding, boolean & arithmetic simplifications
        - Vectorisation
        - Loop unrolling
    - Generates machine code for remaining costly operations

---

# PyPy

Advantages:
- Code still starts straightaway, no initial compilation cost 
- Easy to use - no changes to existing Python code needed
    - just execute with `pypy` instead of with `python`

Disadvantages:
- Only implements core Python functionality
    - e.g. does not support integration with all of NumPy
    - Not compatible with the majority of packages in the Python ecosystem


---

template:titleslide
# Numba

---

# Numba 

- JIT-compiler that uses LLVM toolchain

- Numba lets us select certain functions to be just-in-time (JIT) compiled

- When a Numba function is called for the first time, it is compiled into machine code using LLVM (for given argument types) and stored

- Subsequent calls with same argument types execute the machine code

- Uses built-in version of LLVM, so no calls to external compiler/loading libraries

---

# Numba overview

.center[![:scale_img 90%](numba_flowchart.png)]

[source](https://github.com/ContinuumIO/gtc2017-numba)

---

# Numba example

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

# Numba performance - `nopython` mode

 - A jit-decorated function should run much faster if all of its operations can be compiled into machine code, as there is then no involvement of the (much slower) Python interpreter

- This requires that:
  - all native types in the function can be inferred
  - no new Python objects are created in the function

- If both are true, the Numba-compiled function code does not try to use the interpreter's Python/C API (for type checking or Python object handling)

- `nopython=True` in the `@jit` decorator means the function code will only run if this is the case
 - A warning is issued if these criteria cannot be met, and Numba will fall back to a less optimised mode

---

# Numba performance

 ### Tip: don't 'unvectorise' fast NumPy code!

Numba works well to speed up nonvectorised NumPy code that explicitly iterates over many items

- Does not mean we should rewrite existing vectorised NumPy code to use explicit for loops (slow!) in order for Numba to have a strong effect
  - Any speed up gain from Numba is likely to be cancelled out by removal of vectorised code

---

# Numba performance

Numba works best when:

- The function is called many times during normal execution
 - Compilation is slow, so if the function is not called often, savings in execution time are unlikely to compensate for compilation time
 
- Compute time is primarily due to NumPy array element memory access or numerical operations more complex than a single NumPy function call.

- Function execution time is larger than the Numba dispatcher overhead.
 - Functions that execute in much less than a microsecond are not going to see a major improvement, as the wrapper code which transitions from the Python interpreter to Numba-compiled machine code takes longer than a pure Python function call



---

# PyPy vs Numba

- Numba and PyPy both provide JIT optimising compilation functionality for Python code
 - Numba works *with* the standard Python interpreter, i.e. CPython
 - PyPy is a separate standalone Python compiler implementation

- Numba also does not support usage of every single Numpy feature, but code will still run as normal

---

# Next

- Tomorrow's practical gives you hands on experience speeding up Python code (computational fluid dynamics mini-app) using various strategies covered so far

- Next week's lecture covers parallel computing with Python
  - Will see why multithreading in Python is problematic due to the Global Interpreter Lock (GIL) 

  - Will look at how we can implement thread-based/shared-memory and process-based/distributed-memory parallelism using Numba, Cython, OpenMP and MPI 










