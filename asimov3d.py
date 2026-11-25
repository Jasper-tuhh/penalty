"""
asimov3d.py -- wall-to-wall contact for the 3D monolith / cbox with dolfinx_contact (asimov-contact).

Follows python/demos/demo_nitsche_unbiased.py:
  * unknown is the load-step INCREMENT du, u is the accumulated displacement
  * residual / Jacobian = (your hyperelastic bulk form, evaluated at u + du) + contact_problem contributions
  * contact detection (closest point) is re-run after every accepted step with update_contact_detection(u)
  * dolfinx_contact.NewtonSolver drives the iterations; the adaptive step control (grow / shrink the
    prescribed-displacement increment) is the same logic as driver3d.solve_with_load_stepping.
"""
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import InsertMode, ScatterMode

import ufl
from dolfinx.fem import extract_function_spaces, form
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.mesh import (create_submesh, locate_entities, locate_entities_boundary,
                          meshtags)

from dolfinx_contact.cpp import ContactMode
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.newton_solver import NewtonSolver
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh

CEILING, FLOOR = 1, 2


# ----------------------------------------------------------------------------
# mesh: remove the cavity (no third medium) and tag the two cavity walls
# ----------------------------------------------------------------------------
def body_submesh(mesh, void_marker):
    tdim = mesh.topology.dim
    n_local = mesh.topology.index_map(tdim).size_local
    void = locate_entities(mesh, tdim, void_marker)
    void = void[void < n_local]
    body = np.setdiff1d(np.arange(n_local, dtype=np.int32), void).astype(np.int32)
    return create_submesh(mesh, tdim, body)[0]


def contact_mesh(mesh, x_min, x_max, y_floor=0.1, y_ceiling=0.4, tol=1e-8):
    """Facet tags CEILING (y = 0.4) / FLOOR (y = 0.1) on the cavity walls (x in [x_min, x_max]).
    In parallel the mesh is re-ghosted for the contact search (dolfinx_contact requirement).
    Returns (mesh, facet_marker)."""
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, tdim)
    n_f = mesh.topology.index_map(fdim).size_local
    idx, val = [], []
    for tag, y0 in ((CEILING, y_ceiling), (FLOOR, y_floor)):
        f = locate_entities_boundary(
            mesh, fdim, lambda x, y0=y0: (np.abs(x[1] - y0) < tol)
            & (x[0] >= x_min - tol) & (x[0] <= x_max + tol))
        f = f[f < n_f]
        idx.append(f)
        val.append(np.full(len(f), tag, dtype=np.int32))
    idx, val = np.concatenate(idx), np.concatenate(val)
    order = np.argsort(idx)
    facet_marker = meshtags(mesh, fdim, idx[order].astype(np.int32), val[order])
    if mesh.comm.size > 1:
        n_c = mesh.topology.index_map(tdim).size_local
        cell_marker = meshtags(mesh, tdim, np.arange(n_c, dtype=np.int32),
                               np.ones(n_c, dtype=np.int32))
        mesh, facet_marker, _ = create_contact_mesh(mesh, facet_marker, cell_marker,
                                                    [CEILING, FLOOR], 2.0)
    return mesh, facet_marker


# ----------------------------------------------------------------------------
# adaptive displacement-controlled solve with dolfinx_contact
# ----------------------------------------------------------------------------
def solve_with_asimov(Pi, u, du, bcs, dinc, facet_marker, target, prm, lame, nitsche_param,
                      theta=0.0, q_degree=5, search="closest", emit=None, on_step=None,
                      verbose=False):
    """Pi    : bulk energy written with the total displacement u + du
       u, du : accumulated displacement / increment (Functions on the same space)
       dinc  : Constant used in a Dirichlet BC *of du* (the increment of the prescribed displacement)
       lame  : (mu, lambda) used by the contact kernels (linear-elastic Nitsche terms)
       nitsche_param : parameter handed to generate_contact_data (kernel penalty ~ param / h)
       theta : 0 = penalty-like, 1 symmetric, -1 skew-symmetric Nitsche"""
    V = u.function_space
    mesh = V.mesh
    comm = mesh.comm
    tdim = mesh.topology.dim
    say = emit if emit is not None else (lambda m: None)
    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, 0)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, tdim - 1)
    mesh.topology.create_connectivity(tdim, tdim)

    # ---- bulk forms (no contact terms) ----
    v, w = ufl.TestFunction(V), ufl.TrialFunction(V)
    F_ufl = ufl.derivative(Pi, du, v)
    J_ufl = ufl.derivative(F_ufl, du, w)
    jit = {"cffi_extra_compile_args": [], "cffi_libraries": ["m"]}
    F_c, J_c = form(F_ufl, jit_options=jit), form(J_ufl, jit_options=jit)

    # ---- contact problem (two surfaces of the same body, unbiased) ----
    from dolfinx.fem import Function, functionspace
    V0 = functionspace(mesh, ("DG", 0))
    mu0, lmbda0, fric = Function(V0), Function(V0), Function(V0)
    mu0.interpolate(lambda x: np.full((1, x.shape[1]), lame[0]))
    lmbda0.interpolate(lambda x: np.full((1, x.shape[1]), lame[1]))
    fric.interpolate(lambda x: np.full((1, x.shape[1]), 0.0))          # frictionless
    surfaces = adjacencylist(np.array([CEILING, FLOOR], dtype=np.int32),
                             np.array([0, 2], dtype=np.int32))
    pairs = [(1, 0), (0, 1)]
    mode = ContactMode.Raytracing if search == "raytracing" else ContactMode.ClosestPoint
    contact = ContactProblem([facet_marker], surfaces, pairs, mesh, q_degree, [mode] * len(pairs))
    contact.generate_contact_data(FrictionLaw.Tresca, V,
                                  {"u": u, "du": du, "mu": mu0, "lambda": lmbda0, "fric": fric},
                                  nitsche_param, theta)

    def compute_coefficients(x, coeffs):
        n = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        du.x.array[:n] = x.array_r[:n]
        du.x.scatter_forward()
        contact.update_contact_data(du)

    def compute_residual(x, b, coeffs):
        b.zeroEntries()
        b.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        contact.assemble_vector(b, V)
        assemble_vector(b, F_c)
        apply_lifting(b, [J_c], bcs=[bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=InsertMode.ADD, mode=ScatterMode.REVERSE)
        set_bc(b, bcs, x, -1.0)

    def compute_jacobian(x, a_mat, coeffs):
        a_mat.zeroEntries()
        contact.assemble_matrix(a_mat, V)
        assemble_matrix(a_mat, J_c, bcs=bcs)
        a_mat.assemble()

    a_mat = contact.create_matrix(J_c)
    b = create_vector(extract_function_spaces(F_c))
    newton = NewtonSolver(comm, a_mat, b, contact.coeffs)
    newton.set_residual(compute_residual)
    newton.set_jacobian(compute_jacobian)
    newton.set_coefficients(compute_coefficients)
    newton.set_newton_options({
        "relaxation_parameter": 1.0, "atol": prm["atol"], "rtol": prm["rtol"],
        "convergence_criterion": "residual", "max_it": prm["max_it"],
        "error_on_nonconvergence": False})
    newton.set_krylov_options({"ksp_type": "preonly", "pc_type": "lu",
                               "pc_factor_mat_solver_type": "mumps"})

    # ---- adaptive load stepping (same logic as driver3d.solve_with_load_stepping) ----
    stats = {"newton_iterations": 0, "load_steps": 0, "failed_attempts": 0, "successful_attempts": 0}
    time_snes = 0.0
    initial = abs(target) / prm["Nsteps"]
    min_step = abs(target) / prm["Nsteps_max"]
    max_step = abs(target) / prm["Nsteps_min"]
    step_size, inc = initial, prm["inc"]
    current, n, min_hits, aborted = 0.0, 0, 0, False

    while current > target and not aborted:
        success = False
        for _ in range(prm["max_attempts"]):
            nxt = max(current - step_size, target)
            dinc.value = nxt - current
            try:
                t0 = MPI.Wtime()
                its, converged = newton.solve(du, write_solution=False)
                time_snes += MPI.Wtime() - t0
                if not converged:
                    raise RuntimeError(f"Newton did not converge ({its} its)")
                du.x.scatter_forward()
                u.x.array[:] += du.x.array[:]
                u.x.scatter_forward()
                contact.update_contact_detection(u)
                a_mat = contact.create_matrix(J_c)
                newton.set_petsc_matrix(a_mat)
                du.x.array[:] *= 0.1                      # initial guess of the next step (as in the demo)
                contact.update_contact_data(du)
                stats["newton_iterations"] += its
                stats["successful_attempts"] += 1
                stats["load_steps"] += 1
                n += 1
                current = nxt
                if current <= prm["crit"]:
                    inc = prm["inc_late"]
                step_size = min(step_size * inc, max_step)
                success = True
                if verbose:
                    say(f"  step {n}: disp {current:.5f}, its={its}")
                if on_step is not None:
                    on_step(n, current)
            except Exception as e:
                say(f"Step failed at disp {current:.5f} -> {nxt:.5f}: {e}")
                du.x.array[:] = 0.0
                du.x.scatter_forward()
                contact.update_contact_data(du)
                stats["failed_attempts"] += 1
                step_size = max(step_size * prm["dec"], min_step)
                if step_size == min_step:
                    min_hits += 1
                if min_hits >= 2:
                    say(f"Minimum step size {min_step:.3e} reached and still failing.")
                    aborted = True
            if success or aborted:
                break
        if not success:
            break
    stats.update(current=current, time_snes=time_snes)
    return stats
