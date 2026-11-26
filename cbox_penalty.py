"""
3D "cbox" with wall-to-wall contact from dolfinx_contact (asimov-contact) instead of a third medium.

  * cavity cells are removed (submesh) -> no third medium, no Hessian regularisation
  * ceiling (y = 0.4) / floor (y = 0.1) of the cavity are tagged and contact each other through the
    unbiased closest-point contact of dolfinx_contact.  Bulk = your hyperelastic energy (cbox_3d.py).
  * penalty-like by default (THETA=0).  The contact kernels take (mu, lambda) of a LINEAR-elastic
    material for their Nitsche/consistency terms (here: the small-strain mu and lambda = K - 2/3 mu).
Env (besides driver3d's): PENALTY (1e7), THETA (0), CONTACT_QDEG (5), CONTACT_SEARCH (closest|raytracing),
     NITSCHE_PARAM (overrides the parameter passed to the library; default PENALTY*h)
"""
import os

import numpy as np
from mpi4py import MPI

from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter
from dolfinx.fem import (Constant, Function, dirichletbc, functionspace,
                         locate_dofs_topological, assemble_scalar, form)
from dolfinx.mesh import locate_entities_boundary
from ufl import Measure

from methods import make_material
from functional_3d import kinematics_3d, psi_3d
from driver3d import make_box_mesh, dofs_on_edges, z_support_bcs, sweep_main
from asimov3d import body_submesh, contact_mesh, solve_with_asimov

MATERIAL_MODEL = os.environ.get("MATERIAL_MODEL", "neo_hooke")
CELL_TYPE = "hexahedron"
ZBC = os.environ.get("ZBC", "free")
NZ = int(os.environ.get("NZ", "8"))
WRITE_EVERY = int(os.environ.get("WRITE_EVERY", "10"))
PENALTY = float(os.environ.get("PENALTY", "1e7"))
THETA = float(os.environ.get("THETA", "0.0"))
Q_DEG = int(os.environ.get("CONTACT_QDEG", "5"))
SEARCH = os.environ.get("CONTACT_SEARCH", "closest")
L_LAYER = 0.1
DEPTH = float(os.environ["DEPTH"]) if "DEPTH" in os.environ else 1.0 + L_LAYER
MU, K, GAMMA = 5 / 14, 5 / 3, 1.0e-7
LAME = (MU, K - 2.0 * MU / 3.0)                  # (mu, lambda) for the contact kernels
TARGET = -0.75
PRM = dict(Nsteps=100, Nsteps_min=40, Nsteps_max=800, crit=-0.8, inc=1.5, inc_late=1.1,
           dec=0.5, max_attempts=10,
           max_it=int(os.environ.get("NEWTON_MAXIT", "15")),
           atol=float(os.environ.get("NEWTON_ATOL", "1e-10")),
           rtol=float(os.environ.get("NEWTON_RTOL", "1e-9")))


def run_case(Nx, Ny, fe_degree, reg_method, k_reg, out_dir, emit, verbose=False):
    """reg_method / k_reg are accepted for compatibility with sweep_main and ignored."""
    comm = MPI.COMM_WORLD
    t_start = MPI.Wtime()
    if fe_degree != 1:
        raise ValueError("dolfinx_contact needs displacement degree == geometry degree; the box meshes "
                         "have linear geometry, so use P_VALUES=1 (no Hessian regularisation any more)")
    if Nx % 10 or Ny % 5:
        raise ValueError("cbox needs Nx % 10 == 0 and Ny % 5 == 0 so that the 0.1-wide walls coincide with mesh lines")

    box = make_box_mesh(comm, 1.0, 0.5, DEPTH, Nx, Ny, NZ, CELL_TYPE)
    tol = 1e-8
    body = body_submesh(box, lambda x: (x[0] >= 0.1 - tol) & (x[1] >= 0.1 - tol) & (x[1] <= 0.4 + tol))
    mesh0, facet_marker = contact_mesh(body, 0.1, 1.0)
    tdim = mesh0.topology.dim
    fdim = tdim - 1
    mesh0.topology.create_connectivity(fdim, tdim)
    n_local = mesh0.topology.index_map(tdim).size_local
    h = comm.allreduce(float(np.min(mesh0.h(tdim, np.arange(n_local)))), op=MPI.MIN)

    Vu = functionspace(mesh0, ("Lagrange", fe_degree, (mesh0.geometry.dim,)))
    u = Function(Vu, name="displacement")        # accumulated displacement
    du = Function(Vu, name="increment")          # load-step increment (Newton unknown)
    dx = Measure("dx", domain=mesh0, metadata={"quadrature_degree": 2 * fe_degree + 2})

    # ---- boundary conditions (on the INCREMENT du) ----
    left_facets = locate_entities_boundary(mesh0, fdim, lambda x: np.isclose(x[0], 0.0))
    bcl = dirichletbc(np.zeros(3, dtype=default_scalar_type),
                      locate_dofs_topological(Vu, fdim, left_facets), Vu)
    dinc = Constant(mesh0, default_scalar_type(0.0))      # increment of the prescribed u_y
    corner = lambda x: np.isclose(x[0], 1.0) & np.isclose(x[1], 0.5)
    bct = dirichletbc(dinc, dofs_on_edges(mesh0, Vu.sub(1), corner), Vu.sub(1))
    bcs = [bcl, bct] + z_support_bcs(mesh0, Vu, DEPTH, ZBC)

    # ---- bulk energy at u + du ----
    mat = make_material(MU, K, MATERIAL_MODEL)
    Pi = psi_3d(mat, kinematics_3d(u + du)) * dx
    nitsche_param = float(os.environ.get("NITSCHE_PARAM", PENALTY * h))

    dof_count = Vu.dofmap.index_map.size_global * Vu.dofmap.index_map_bs
    num_cells = mesh0.topology.index_map(tdim).size_global
    emit(f"--- cbox asimov-contact p={fe_degree}, {Nx}x{Ny}x{NZ} {CELL_TYPE} (cavity removed), "
         f"{num_cells} cells, {dof_count} dofs, {MATERIAL_MODEL}, ZBC={ZBC}, depth={DEPTH:g}, "
         f"param={nitsche_param:.3e} (~penalty {PENALTY:.1e}), theta={THETA:g}, "
         f"{len(facet_marker.find(1))}+{len(facet_marker.find(2))} wall facets (local) ---")
    time_setup = MPI.Wtime() - t_start

    writer = VTXWriter(comm, os.path.join(out_dir, "results_u.bp"), [u], engine="BP4")
    try:
        writer.write(0.0)
        stats = solve_with_asimov(
            Pi, u, du, bcs, dinc, facet_marker, TARGET, PRM, LAME, nitsche_param,
            theta=THETA, q_degree=Q_DEG, search=SEARCH, emit=emit, verbose=verbose,
            on_step=lambda n, d: writer.write(float(n)) if n % WRITE_EVERY == 0 else None)
        writer.write(float(stats["load_steps"]) + 0.5)       # final state
    finally:
        writer.close()

    # strain energy of the bulk at the final state (contact energy is not exposed by the library)
    Pi_u = psi_3d(mat, kinematics_3d(u)) * dx
    potential = comm.allreduce(assemble_scalar(form(Pi_u)), op=MPI.SUM)
    steps = stats["load_steps"]
    return {
        "fe_degree": fe_degree, "Nx": Nx, "Ny": Ny, "Nz": NZ, "mesh_h": h,
        "dof": int(dof_count), "num_cells": int(num_cells),
        "final_displacement": stats["current"], "n_steps": steps, "potential": potential,
        "time_total": comm.allreduce(MPI.Wtime() - t_start, op=MPI.MAX),
        "time_setup": comm.allreduce(time_setup, op=MPI.MAX),
        "time_snes": comm.allreduce(stats["time_snes"], op=MPI.MAX),
        "newton_iterations": stats["newton_iterations"],
        "avg_newton_per_step": stats["newton_iterations"] / steps if steps else 0.0,
        "failed_attempts": stats["failed_attempts"],
        "successful_attempts": stats["successful_attempts"],
        "disp_pct": stats["current"] / TARGET * 100.0,
        "u": u, "Vu": Vu, "mesh": mesh0, "material_tags": None,
    }


if __name__ == "__main__":
    sweep_main("cbox_penalty", run_case, default_meshes="20x10,30x15,40x20")
