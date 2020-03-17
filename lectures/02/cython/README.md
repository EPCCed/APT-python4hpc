
# Cython example

This example shows how to use Cython and what performance gains can be had

`fib.py` - the original pure Python code
`fib_python.pyx` - the same original pure Python code (no static C-type declarations), saved as `.pyx` ready to cythonize into C and then compile into a shared library
`fib_cython.pyx` - Cython version of the code (Python plus static C-type declarations), saved as `.pyx` ready to cythonize into C and then compile into a shared library

To create the shared libraries, run the following commands:

`python setup_python.py build_ext --inplace`

`python setup_cython.py build_ext --inplace`

This will create the corresponding `.so` library files

It is worth inspecting the corresponding `.c` source code files generated to see what the Cython compiler has
created - you can see these CPython extension modules are quite involved even for relatively simple code.

To run and obtain timings in ipython:

```Python
> import fib
> fib?
Type:        module
String form: <module 'fib' from 'fib.py'>
>
> %timeit fib.fib(200)
9.21 µs ± 225 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
>
>
> import fib_python
> fib_python?
Type:        module
String form: <module 'fib_python' from 'fib_python.cpython.so'>
>
> %timeit fib_python.fib(200)
1.38 µs ± 13.7 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
>
>
> import fib_cython
> fib_cython?
Type:        module
String form: <module 'fib_cython' from 'fib_cython.cpython.so'>
>
> %timeit fib_cython.fib(200)
245 ns ± 4.17 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```


 Implementation           | Runtime | Speedup
--------------------------|---------|--------
 Pure Python              | 9.21 µs |    1 `x`
 Cythonized Pure Python   | 1.38 µs |    7 `x`
 Cythonized Cython        |  245 ns |  37  `x`

