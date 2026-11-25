#!/usr/bin/env python
# coding: utf-8
"""
Hertzian contact with a rigid indenter using a penalty approach (FEniCSx / dolfinx).

Targets dolfinx >= 0.10 (SNES-based NonlinearProblem).

Analytical Hertz solution (parabolic indenter of radius R, depth d):
    a  = sqrt(R d)
    F  = 4/3 * E/(1-nu^2) * a * d
    p0 = 3F / (2 pi a^2)
"""
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh as dmesh, io
from dolfinx.fem import petsc as fem_petsc

comm = MPI.COMM_WORLD

# ---------------------------------------------------------------------------
# Mesh: unit cube of hexahedra, refined near x = y = z = 0, mapped to
# [0,1] x [0,1] x [-1,0]
# ---------------------------------------------------------------------------
N = 30
mesh = dmesh.create_unit_cube(comm, N, N, N // 2, dmesh.CellType.hexahedron)

coords = mesh.geometry.x  # modified in place (includes ghost nodes consistently)
coords[:, :2] = coords[:, :2] ** 2
coords[:, 2] = -coords[:, 2] ** 2

# ---------------------------------------------------------------------------
# Boundary facets
# ---------------------------------------------------------------------------
tdim = mesh.topology.dim
fdim = tdim - 1


def top(x):
    return np.isclose(x[2], 0.0)


def symmetry_x(x):
    return np.isclose(x[0], 0.0)


def symmetry_y(x):
    return np.isclose(x[1], 0.0)


def bottom(x):
    return np.isclose(x[2], -1.0)


top_facets = dmesh.locate_entities_boundary(mesh, fdim, top)
sym_x_facets = dmesh.locate_entities_boundary(mesh, fdim, symmetry_x)
sym_y_facets = dmesh.locate_entities_boundary(mesh, fdim, symmetry_y)
bottom_facets = dmesh.locate_entities_boundary(mesh, fdim, bottom)

# Mark the top surface with tag 1 (replaces MeshFunction + SubDomain.mark)
order = np.argsort(top_facets)
facet_tags = dmesh.meshtags(
    mesh, fdim, top_facets[order], np.full(len(top_facets), 1, dtype=np.int32)
)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

# ---------------------------------------------------------------------------
# Obstacle, function spaces, unknowns
# ---------------------------------------------------------------------------
R = 0.5
d = 0.02

x = ufl.SpatialCoordinate(mesh)
obstacle = -d + (x[0] ** 2 + x[1] ** 2) / (2 * R)

V = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))  # displacement
V2 = fem.functionspace(mesh, ("Lagrange", 1))                       # gap
V0 = fem.functionspace(mesh, ("DG", 0))                             # pressure

u = fem.Function(V, name="Displacement")
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
gap = fem.Function(V2, name="Gap")
p = fem.Function(V0, name="Contact pressure")

# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
u_zero = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
bc_bottom = fem.dirichletbc(
    u_zero, fem.locate_dofs_topological(V, fdim, bottom_facets), V
)
bc_x = fem.dirichletbc(
    PETSc.ScalarType(0.0),
    fem.locate_dofs_topological(V.sub(0), fdim, sym_x_facets),
    V.sub(0),
)
bc_y = fem.dirichletbc(
    PETSc.ScalarType(0.0),
    fem.locate_dofs_topological(V.sub(1), fdim, sym_y_facets),
    V.sub(1),
)
bcs = [bc_bottom, bc_x, bc_y]

# ---------------------------------------------------------------------------
# Material and weak form
# ---------------------------------------------------------------------------
E = 10.0
nu = 0.3
mu = E / 2 / (1 + nu)
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)


def eps(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(eps(v)) * ufl.Identity(3) + 2.0 * mu * eps(v)


def ppos(x):
    return (x + abs(x)) / 2.0


pen = fem.Constant(mesh, PETSc.ScalarType(1e4))
ds_top = ds(1, metadata={"quadrature_degree": 4})

form = (
    ufl.inner(sigma(u), eps(u_)) * ufl.dx
    + pen * ufl.dot(u_[2], ppos(u[2] - obstacle)) * ds_top
)
J = ufl.derivative(form, u, du)

# ---------------------------------------------------------------------------
# Nonlinear solver (PETSc SNES Newton + CG)
# ---------------------------------------------------------------------------
# ILU is serial-only in PETSc; use GAMG when running with MPI.
petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_rtol": 1e-8,
    "snes_atol": 1e-10,
    "snes_max_it": 50,
    "snes_monitor": None,
    "ksp_type": "cg",
    "ksp_rtol": 1e-8,
    "pc_type": "ilu" if comm.size == 1 else "gamg",
}

problem = fem_petsc.NonlinearProblem(
    form,
    u,
    bcs=bcs,
    J=J,
    petsc_options_prefix="contact_",
    petsc_options=petsc_options,
)
problem.solve()
u.x.scatter_forward()

reason = problem.solver.getConvergedReason()
n_iter = problem.solver.getIterationNumber()
if comm.rank == 0:
    print(f"Newton: converged reason={reason} ({'OK' if reason > 0 else 'FAILED'}) in {n_iter} iterations")

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
# Pressure p = -sigma_zz, evaluated at cell midpoints (DG0 interpolation points)
p_expr = fem.Expression(-sigma(u)[2, 2], V0.element.interpolation_points)
p.interpolate(p_expr)

# Gap g = h - u_z (nodal values)
gap_expr = fem.Expression(obstacle - u[2], V2.element.interpolation_points)
gap.interpolate(gap_expr)

a = np.sqrt(R * d)
F = 4 / 3.0 * E / (1 - nu**2) * a * d
p0 = 3 * F / (2 * np.pi * a**2)

p_max = comm.allreduce(np.max(np.abs(p.x.array)), op=MPI.MAX)
# factor 4 because of the double symmetry of the problem
F_fe = 4 * comm.allreduce(
    fem.assemble_scalar(fem.form(p * ds(1))), op=MPI.SUM
)

if comm.rank == 0:
    print(f"Maximum pressure FE: {p_max:8.5f} Hertz: {p0:8.5f}")
    print(f"Applied force    FE: {F_fe:8.5f} Hertz: {F:8.5f}")

with io.XDMFFile(comm, "contact_penalty_results.xdmf", "w") as f:
    f.write_mesh(mesh)
    f.write_function(u, 0.0)
    f.write_function(gap, 0.0)
    f.write_function(p, 0.0)