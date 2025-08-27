import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

# ----------------------------------------------------------------------
# Portable, high-performance solver selection
#  - x86_64 (Intel/AMD): try PARDISO (pypardiso or legacy pyMKL)
#  - Apple Silicon / other arch: use SciPy; prefer UMFPACK if available
#  - Allow manual override via env var: ANGLER_SOLVER={pardiso,scipy}
# ----------------------------------------------------------------------
import os
import platform

_ARCH = platform.machine().lower()
_IS_X86_64 = _ARCH in ("x86_64", "amd64")
_ENV_SOLVER = os.getenv("ANGLER_SOLVER", "").lower()

HAVE_PYPARDISO = False
HAVE_PYMKL = False
HAVE_UMFPACK = False

if _IS_X86_64:
    try:
        # Preferred modern wrapper; bundles MKL RT on common platforms
        from pypardiso import spsolve as _pardiso_spsolve
        HAVE_PYPARDISO = True
    except Exception:
        pass
    try:
        # Legacy interface used by older code
        from pyMKL import pardisoSolver as _pyMKL_pardisoSolver
        HAVE_PYMKL = True
    except Exception:
        pass

try:
    # If installed, SciPy can route to UMFPACK for speed on CSC matrices
    import scikits.umfpack as _umf
    HAVE_UMFPACK = True
except Exception:
    HAVE_UMFPACK = False

# Default solver policy (can be overridden by angler.constants.DEFAULT_SOLVER)
if _ENV_SOLVER in ("pardiso", "pypardiso"):
    SOLVER = "pardiso" if _IS_X86_64 and (HAVE_PYPARDISO or HAVE_PYMKL) \
        else "scipy"
else:
    SOLVER = "scipy"

from time import time

from angler.constants import DEFAULT_MATRIX_FORMAT, DEFAULT_SOLVER
from angler.constants import EPSILON_0, MU_0
from angler.pml import S_create
from angler.derivatives import createDws


def grid_average(center_array, w):
    # computes values at cell edges

    xy = {'x': 0, 'y': 1}
    center_shifted = np.roll(center_array, 1, axis=xy[w])
    avg_array = (center_shifted+center_array)/2
    return avg_array


def dL(N, xrange, yrange=None):
    # solves for the grid spacing

    if yrange is None:
        L = np.array([np.diff(xrange)[0]])  # Simulation domain lengths
    else:
        L = np.array([np.diff(xrange)[0],
                      np.diff(yrange)[0]])  # Simulation domain lengths
    return L/N


def is_equal(matrix1, matrix2):
    # checks if two sparse matrices are equal

    return (matrix1 != matrix2).nnz == 0


def construct_A(omega, xrange, yrange, eps_r, NPML, pol, L0,
                averaging=True,
                timing=False,
                matrix_format=DEFAULT_MATRIX_FORMAT):
    # makes the A matrix
    N = np.asarray(eps_r.shape)  # Number of mesh cells
    M = np.prod(N)  # Number of unknowns

    EPSILON_0_ = EPSILON_0*L0
    MU_0_ = MU_0*L0

    if pol == 'Ez':
        vector_eps_z = EPSILON_0_*eps_r.reshape((-1,))
        T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format=matrix_format)

        (Sxf, Sxb, Syf, Syb) = S_create(omega, L0, N, NPML, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))

        A = (Dxf*1/MU_0_).dot(Dxb) \
            + (Dyf*1/MU_0_).dot(Dyb) \
            + omega**2*T_eps_z

    elif pol == 'Hz':
        if averaging:
            vector_eps_x = grid_average(EPSILON_0_*eps_r, 'x').reshape((-1,))
            vector_eps_y = grid_average(EPSILON_0_*eps_r, 'y').reshape((-1,))
        else:
            vector_eps_x = EPSILON_0_*eps_r.reshape((-1,))
            vector_eps_y = EPSILON_0_*eps_r.reshape((-1,))

        # Setup the T_eps_x, T_eps_y, T_eps_x_inv, and T_eps_y_inv matrices
        T_eps_x = sp.spdiags(vector_eps_x, 0, M, M, format=matrix_format)
        T_eps_y = sp.spdiags(vector_eps_y, 0, M, M, format=matrix_format)
        T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M, format=matrix_format)
        T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M, format=matrix_format)

        (Sxf, Sxb, Syf, Syb) = S_create(omega, L0, N, NPML, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))

        A =   Dxf.dot(T_eps_x_inv).dot(Dxb) \
            + Dyf.dot(T_eps_y_inv).dot(Dyb) \
            + omega**2*MU_0_*sp.eye(M)

    else:
        raise ValueError("something went wrong and pol is not one of Ez, Hz, instead was given {}".format(pol))

    derivs = {
        'Dyb' : Dyb,
        'Dxb' : Dxb,
        'Dxf' : Dxf,
        'Dyf' : Dyf
    }

    return (A, derivs)


def solver_eigs(A, Neigs, guess_value=0, guess_vector=None, timing=False):
    # solves for the eigenmodes of A

    if timing:
        start = time()
    (values, vectors) = spl.eigs(A, k=Neigs, sigma=guess_value, v0=guess_vector, which='LM')
    if timing:
        end = time()
        print('Elapsed time for eigs() is %.4f secs' % (end - start))
    return (values, vectors)


def solver_direct(A, b, timing=False, solver=SOLVER):
    """
    Solve A x = b. Fast path prefers PARDISO on x86_64. Otherwise uses
    SciPy, ensuring CSC format and leveraging UMFPACK if available.

    Args:
        A: sparse matrix (prefer CSR/CSC), complex OK.
        b: rhs vector/array (complex supported).
        timing: print solve time.
        solver: 'pardiso' or 'scipy'. Auto-selected at import, but can
                be overridden per-call.

    Returns:
        x: solution vector (np.ndarray).
    """
    b = b.astype(np.complex128).reshape((-1,))
    if not b.any():
        return np.zeros(b.shape, dtype=np.complex128)

    if timing:
        t = time()

    # Convert to CSC for best performance in both UMFPACK and SuperLU
    A_csc = A if sp.isspmatrix_csc(A) else A.tocsc()

    if solver.lower() == "pardiso":
        if HAVE_PYPARDISO:
            # pypardiso has a simple spsolve-style API
            x = _pardiso_spsolve(A_csc, b)
        elif HAVE_PYMKL:
            # Legacy pyMKL path with explicit factor/solve/clear
            # 13 = complex, unsymmetric (SC-PML makes it non-Hermitian)
            ps = _pyMKL_pardisoSolver(A_csc, mtype=13)
            ps.factor()
            x = ps.solve(b)
            ps.clear()
        else:
            # If user forced 'pardiso' but none is present, fall back
            x = spl.spsolve(A_csc, b)
    elif solver.lower() == "scipy":
        # Try to request UMFPACK when available (SciPy may accept kw)
        try:
            x = spl.spsolve(A_csc, b, use_umfpack=HAVE_UMFPACK)
        except TypeError:
            # SciPy without 'use_umfpack' arg
            x = spl.spsolve(A_csc, b)
    else:
        raise ValueError(
            "Invalid solver choice: {} (use 'pardiso' or 'scipy')"
            .format(str(solver))
        )

    if timing:
        print("Linear system solve took {:.2f} seconds".format(time() - t))

    return x


def solver_complex2real(A11, A12, b, timing=False, solver=SOLVER):
    """
    Solve the real-embedded system:
      [A11, A12; A21*, A22*] [x; x*] = [b; b*]
    built as a 2N x 2N real system.

    Notes:
        - Real unsymmetric system (PARDISO mtype=11).
        - We assemble CSC and apply same solver policy as solver_direct.
    """
    b = b.astype(np.complex128).reshape((-1,))
    N = b.size
    if not b.any():
        return np.zeros(b.shape, dtype=np.complex128)

    b_re = np.real(b).astype(np.float64)
    b_im = np.imag(b).astype(np.float64)

    # Build the 2N x 2N real system in block form
    Areal = sp.vstack((
        sp.hstack((np.real(A11) + np.real(A12),
                   -np.imag(A11) + np.imag(A12))),
        sp.hstack((np.imag(A11) + np.imag(A12),
                   np.real(A11) - np.real(A12)))
    ))
    A_csc = Areal if sp.isspmatrix_csc(Areal) else Areal.tocsc()
    rhs = np.hstack((b_re, b_im))

    if timing:
        t = time()

    if solver.lower() == "pardiso":
        if HAVE_PYPARDISO:
            x = _pardiso_spsolve(A_csc, rhs)
        elif HAVE_PYMKL:
            # 11 = real unsymmetric
            ps = _pyMKL_pardisoSolver(A_csc, mtype=11)
            ps.factor()
            x = ps.solve(rhs)
            ps.clear()
        else:
            x = spl.spsolve(A_csc, rhs)
    elif solver.lower() == "scipy":
        try:
            x = spl.spsolve(A_csc, rhs, use_umfpack=HAVE_UMFPACK)
        except TypeError:
            x = spl.spsolve(A_csc, rhs)
    else:
        raise ValueError(
            "Invalid solver choice: {} (use 'pardiso' or 'scipy')"
            .format(str(solver))
        )

    if timing:
        print("Linear system solve took {:.2f} seconds".format(time() - t))

    # Recombine to complex solution
    return x[:N] + 1j * x[N:2*N]