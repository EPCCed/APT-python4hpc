template: titleslide

# Introduction to Performance Programming with Python
## 2

---

# ...where were we?

Performance implications of standard Python interpreter:
- Overheads from type checking & stack-based execution through VM
- Limited optimisations during compilation into Python bytecode

Fast numerical computing:
- Use NumPy arrays (statically typed, fixed size)
- Use efficient array access syntax to minimise creation of temporaries
- Don't use explicit `for` loops that iterate over NumPy array elements
- Use overloaded array operators (`+`, `-`, `*`, `/`, `**`) and other [`ufuncs`](https://numpy.org/devdocs/reference/ufuncs.html)
  - If available, try dedicated functions provided by NumPy, SciPy, etc.
- NumExpr can speed up array expressions by minimising temporaries and optimising cache usage
- Interface Python with Fortran/C/C++ code compiled into machine code by importing functions from shared library (many options, looked at `ctypes`)
  
---

# What's next?

4 other strategies for speeding up Python performance:

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

CPython and NumPy provide a C API, allows you to:
  - Write C code (or C++ with restrictions) that extends CPython 
  - Operate on CPython's underlying C data types directly
    - including NumPy arrays
  - Import as Python module after compiling into shared library 

`my_extension.c`:
```C
#include "Python.h"
#include "arrayobject.h"

// Map Python objects (NumPy arrays) to C, and initialise

// Declare function calls that will be accessible from Python

// C code for performance-critical operations
//   Functions return Python object(s) (e.g. NumPy), if applicable
```

---

# CPython extension modules

- Typical usage:
 - Accelerator module: self-contained C code, for speed
 - Wrapper module: interface with other C code (libraries)
 - Low-level system access: interface with OS, hardware, ...
  
- NumPy extension modules:
 - Accelerator modules: e.g. inner loops moved to C
 - Wrapper modules: e.g. call Fortran/C linear algebra libraries
 - Low-level: e.g. vectorise operations, control memory layout of arrays

---

# CPython extension modules

- Advantage: extremely flexible

- Disadvantages:
    - May require thorough understanding of CPython and its Python/C API
        - Significant (initial) development effort
        - May be difficult to maintain (C API changes)

- Cython (next) partly automates creation of CPython extension modules
    - Does not require intimate knowledge of CPython
    - Expect Cython to take care of changes in CPython

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

- When a JIT-decorated function is called for the first time, it is compiled into machine code using LLVM (for given argument types) 

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

- Both easy to use:
    - Numba uses simple decorators
    - No code changes needed for PyPy

- Compatibility:
    - Numba works with the standard Python interpreter, i.e. CPython and NumPy
        - not every single Numpy feature, but code will still run as normal
    - PyPy support for NumPy, SciPy, pandas etc. still relatively immature

- Performance:
    - It depends... try it!
    - Numba more generally useful for numerical computing

- Numba can also be used to target GPUs

---

# Next

- Tomorrow's practical gives you hands on experience speeding up Python code (computational fluid dynamics mini-app) using various strategies covered so far

- Next week's lecture covers parallel computing with Python
  - Will see why multithreading in Python is problematic due to the Global Interpreter Lock (GIL) 

  - Will look at how we can implement thread-based/shared-memory and process-based/distributed-memory parallelism using Numba, Cython, OpenMP and MPI 










