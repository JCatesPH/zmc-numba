*> \brief <b> CGTSV computes the solution to system of linear equations A * X = B for GT matrices </b>
 *
 *  =========== DOCUMENTATION ===========
 *
 * Online html documentation available at
 *            http://www.netlib.org/lapack/explore-html/
 *
 *> \htmlonly
 *> Download CGTSV + dependencies
 *> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/cgtsv.f">
 *> [TGZ]</a>
 *> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/cgtsv.f">
 *> [ZIP]</a>
 *> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/cgtsv.f">
 *> [TXT]</a>
 *> \endhtmlonly
 *
 *  Definition:
 *  ===========
 *
 *       SUBROUTINE CGTSV( N, NRHS, DL, D, DU, B, LDB, INFO )
 *
 *       .. Scalar Arguments ..
 *       INTEGER            INFO, LDB, N, NRHS
 *       ..
 *       .. Array Arguments ..
 *       COMPLEX            B( LDB, * ), D( * ), DL( * ), DU( * )
 *       ..
 *
 *
 *> \par Purpose:
 *  =============
 *>
 *> \verbatim
 *>
 *> CGTSV  solves the equation
 *>
 *>    A*X = B,
 *>
 *> where A is an N-by-N tridiagonal matrix, by Gaussian elimination with
 *> partial pivoting.
 *>
 *> Note that the equation  A**T *X = B  may be solved by interchanging the
 *> order of the arguments DU and DL.
 *> \endverbatim
 *
 *  Arguments:
 *  ==========
 *
 *> \param[in] N
 *> \verbatim
 *>          N is INTEGER
 *>          The order of the matrix A.  N >= 0.
 *> \endverbatim
 *>
 *> \param[in] NRHS
 *> \verbatim
 *>          NRHS is INTEGER
 *>          The number of right hand sides, i.e., the number of columns
 *>          of the matrix B.  NRHS >= 0.
 *> \endverbatim
 *>
 *> \param[in,out] DL
 *> \verbatim
 *>          DL is COMPLEX array, dimension (N-1)
 *>          On entry, DL must contain the (n-1) subdiagonal elements of
 *>          A.
 *>          On exit, DL is overwritten by the (n-2) elements of the
 *>          second superdiagonal of the upper triangular matrix U from
 *>          the LU factorization of A, in DL(1), ..., DL(n-2).
 *> \endverbatim
 *>
 *> \param[in,out] D
 *> \verbatim
 *>          D is COMPLEX array, dimension (N)
 *>          On entry, D must contain the diagonal elements of A.
 *>          On exit, D is overwritten by the n diagonal elements of U.
 *> \endverbatim
 *>
 *> \param[in,out] DU
 *> \verbatim
 *>          DU is COMPLEX array, dimension (N-1)
 *>          On entry, DU must contain the (n-1) superdiagonal elements
 *>          of A.
 *>          On exit, DU is overwritten by the (n-1) elements of the first
 *>          superdiagonal of U.
 *> \endverbatim
 *>
 *> \param[in,out] B
 *> \verbatim
 *>          B is COMPLEX array, dimension (LDB,NRHS)
 *>          On entry, the N-by-NRHS right hand side matrix B.
 *>          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
 *> \endverbatim
 *>
 *> \param[in] LDB
 *> \verbatim
 *>          LDB is INTEGER
 *>          The leading dimension of the array B.  LDB >= max(1,N).
 *> \endverbatim
 *>
 *> \param[out] INFO
 *> \verbatim
 *>          INFO is INTEGER
 *>          = 0:  successful exit
 *>          < 0:  if INFO = -i, the i-th argument had an illegal value
 *>          > 0:  if INFO = i, U(i,i) is exactly zero, and the solution
 *>                has not been computed.  The factorization has not been
 *>                completed unless i = N.
 *> \endverbatim
 *
 *  Authors:
 *  ========
 *
 *> \author Univ. of Tennessee
 *> \author Univ. of California Berkeley
 *> \author Univ. of Colorado Denver
 *> \author NAG Ltd.
 *
 *> \date December 2016
 *
 *> \ingroup complexGTsolve
 *
 *  =====================================================================
       SUBROUTINE cgtsv( N, NRHS, DL, D, DU, B, LDB, INFO )
 *
 *  -- LAPACK driver routine (version 3.7.0) --
 *  -- LAPACK is a software package provided by Univ. of Tennessee,    --
 *  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
 *     December 2016
 *
 *     .. Scalar Arguments ..
       INTEGER            INFO, LDB, N, NRHS
 *     ..
 *     .. Array Arguments ..
       COMPLEX            B( ldb, * ), D( * ), DL( * ), DU( * )
 *     ..
 *
 *  =====================================================================
 *
 *     .. Parameters ..
       COMPLEX            ZERO
       parameter( zero = ( 0.0e+0, 0.0e+0 ) )
 *     ..
 *     .. Local Scalars ..
       INTEGER            J, K
       COMPLEX            MULT, TEMP, ZDUM
 *     ..
 *     .. Intrinsic Functions ..
       INTRINSIC          abs, aimag, max, real
 *     ..
 *     .. External Subroutines ..
       EXTERNAL           xerbla
 *     ..
 *     .. Statement Functions ..
       REAL               CABS1
 *     ..
 *     .. Statement Function definitions ..
       cabs1( zdum ) = abs( REAL( ZDUM ) ) + abs( AIMAG( zdum ) )
 *     ..
 *     .. Executable Statements ..
 *
       info = 0
       IF( n.LT.0 ) THEN
          info = -1
       ELSE IF( nrhs.LT.0 ) THEN
          info = -2
       ELSE IF( ldb.LT.max( 1, n ) ) THEN
          info = -7
       END IF
       IF( info.NE.0 ) THEN
          CALL xerbla( 'CGTSV ', -info )
          RETURN
       END IF
 *
       IF( n.EQ.0 )
      $   RETURN
 *
       DO 30 k = 1, n - 1
          IF( dl( k ).EQ.zero ) THEN
 *
 *           Subdiagonal is zero, no elimination is required.
 *
             IF( d( k ).EQ.zero ) THEN
 *
 *              Diagonal is zero: set INFO = K and return; a unique
 *              solution can not be found.
 *
                info = k
                RETURN
             END IF
          ELSE IF( cabs1( d( k ) ).GE.cabs1( dl( k ) ) ) THEN
 *
 *           No row interchange required
 *
             mult = dl( k ) / d( k )
             d( k+1 ) = d( k+1 ) - mult*du( k )
             DO 10 j = 1, nrhs
                b( k+1, j ) = b( k+1, j ) - mult*b( k, j )
    10       CONTINUE
             IF( k.LT.( n-1 ) )
      $         dl( k ) = zero
          ELSE
 *
 *           Interchange rows K and K+1
 *
             mult = d( k ) / dl( k )
             d( k ) = dl( k )
             temp = d( k+1 )
             d( k+1 ) = du( k ) - mult*temp
             IF( k.LT.( n-1 ) ) THEN
                dl( k ) = du( k+1 )
                du( k+1 ) = -mult*dl( k )
             END IF
             du( k ) = temp
             DO 20 j = 1, nrhs
                temp = b( k, j )
                b( k, j ) = b( k+1, j )
                b( k+1, j ) = temp - mult*b( k+1, j )
    20       CONTINUE
          END IF
    30 CONTINUE
       IF( d( n ).EQ.zero ) THEN
          info = n
          RETURN
       END IF
 *
 *     Back solve with the matrix U from the factorization.
 *
       DO 50 j = 1, nrhs
          b( n, j ) = b( n, j ) / d( n )
          IF( n.GT.1 )
      $      b( n-1, j ) = ( b( n-1, j )-du( n-1 )*b( n, j ) ) / d( n-1 )
          DO 40 k = n - 2, 1, -1
             b( k, j ) = ( b( k, j )-du( k )*b( k+1, j )-dl( k )*
      $                  b( k+2, j ) ) / d( k )
    40    CONTINUE
    50 CONTINUE
 *
       RETURN
 *
 *     End of CGTSV
 *
       END