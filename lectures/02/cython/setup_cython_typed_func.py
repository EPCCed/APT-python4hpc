from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fib_cython_typed_func.pyx", annotate=True),
)
