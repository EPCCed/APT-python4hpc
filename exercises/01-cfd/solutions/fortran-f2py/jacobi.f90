! 
! Jacobi routine for CFD calculation
!
subroutine jacobi(m, n, niter, psi)
    implicit none

    integer, intent(in) :: m, n, niter
    real*8, dimension(0:m+1, 0:n+1), intent(inout) :: psi
    !f2py intent(in) m
    !f2py intent(in) n
    !f2py intent(in) niter
    !f2py intent(inplace) psi

    integer :: iter, i, j
    real*8, dimension(0:m+1, 0:n+1) :: tmp

    tmp = 0.0

    write(*,*) "niter = ", niter

    do iter = 1, niter
       do i = 1, m
          do j = 1, n
             tmp(j, i) = 0.25 * (psi(j+1,i) + psi(j-1,i) + psi(j,i+1) + psi(j,i-1))
          end do
       end do
       psi(1:m,1:n) = tmp(1:m,1:n)
    end do

end subroutine jacobi

