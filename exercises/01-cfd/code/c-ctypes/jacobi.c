/* 
 * Jacobi iteration
 *
 * m+2     m is physical number of points x-direction
 * n+2     n is physical number of points y-direction
 * niter   number of jacobi iterations
 * psi     stream function expected to be of extent (m+2)*(n+2)
 *
 * Returns 0 on success.
 */

#include <stdlib.h>

int jacobi(int m, int n, int niter, double * psi) {

  int iter;
  int i, j;
  double * tmp = NULL;

  /* Temporary space of extent m*n for the update */

  tmp = (double *) calloc(m*n, sizeof(double));
  if (tmp == NULL) return -1;

  for (iter = 0; iter < niter; iter++) {
    /* Compute differences in true domain 1..m 1..n */
    for (i = 1; i <= (m-2); i++) {
      for (j = 1; j <= (n-2); j++)  {
	int im1 = (i-1)*n + j;
	int ip1 = (i+1)*n + j;
	int jm1 = i*n + j - 1;
	int jp1 = i*n + j + 1;
	tmp[i*n + j] = 0.25*(psi[im1] + psi[ip1] + psi[jm1] + psi[jp1]);
      }
    }

    /* Update */
    for (i = 1; i <= (m-2); i++)  {
      for (j = 1; j <= (n-2); j++)  {
	psi[i*n + j] = tmp[i*n + j];
      }
    }
  }

  free(tmp);

  return 0;
} 

