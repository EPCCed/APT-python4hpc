cdef double fib(int n):
  cdef int i
  cdef double a = 0.0
  cdef double b = 1.0
  for i in range(n):
    a, b = a + b, a
  return a

