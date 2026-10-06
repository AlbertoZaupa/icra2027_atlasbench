import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from utils import condense_input_constrained, dare, get_parametric_qp_terms


ROOT = Path(__file__).resolve().parent
HORIZON = 30
NSTEPS = 300


@dataclass
class Problem:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    x0: np.ndarray
    N: int = HORIZON

    @classmethod
    def load(cls):
        with np.load(ROOT / "data" / "atlas.npz") as data:
            A, B = data["A"], data["B"]
            Q, R = np.diag(data["Q"]), np.diag(data["R"])
            return cls(A, B, Q, R, dare(A, B, Q, R),
                       -data["d_u"] - data["u_ref"],
                       data["d_u"] - data["u_ref"],
                       data["x0"] - data["x_ref"])


class DaocpSolver:
    def __init__(self, problem):
        from daocp import OCPsolver

        self.solver = OCPsolver(
            A=problem.A, B=problem.B,
            Q=[problem.Q.copy() for _ in range(problem.N)] + [problem.P.copy()],
            R=problem.R, lbu=problem.lower, ubu=problem.upper,
            x0=problem.x0, N=problem.N, greedy=False,
        )

    def step(self, state):
        self.solver.update(state)
        start = perf_counter_ns()
        result = self.solver.solve()
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        if result.info.status != "SOLVED":
            raise RuntimeError(f"DAOCP failed: {result.info.status}")
        return np.asarray(result.u[0]).reshape(-1), elapsed_ms, result.info.iters


class HpipmSolver:
    def __init__(self, problem):
        from hpipm_python.common import (
            hpipm_ocp_qp, hpipm_ocp_qp_dim, hpipm_ocp_qp_sol,
            hpipm_ocp_qp_solver, hpipm_ocp_qp_solver_arg,
        )

        nx, nu = problem.B.shape
        N = problem.N
        dim = hpipm_ocp_qp_dim(N)
        dim.set("nx", nx, 0, N)
        dim.set("nu", nu, 0, N - 1)
        dim.set("nbx", nx, 0)
        dim.set("nbu", nu, 0, N - 1)
        self.qp = hpipm_ocp_qp(dim)
        # Preserve the original terminal coordinate change, using the common DARE P.
        eigenvalues, eigenvectors = np.linalg.eigh(problem.P)
        self.qp.set("A", problem.A, 0, N - 2)
        self.qp.set("B", problem.B, 0, N - 2)
        self.qp.set("A", eigenvectors.T @ problem.A, N - 1)
        self.qp.set("B", eigenvectors.T @ problem.B, N - 1)
        self.qp.set("Q", problem.Q, 0, N - 1)
        self.qp.set("Q", np.diag(eigenvalues), N)
        self.qp.set("R", problem.R, 0, N - 1)
        self.qp.set("S", np.zeros((nu, nx)), 0, N - 1)
        self.qp.set("q", np.zeros(nx), 0, N)
        self.qp.set("r", np.zeros(nu), 0, N - 1)
        self.qp.set("Ju", np.eye(nu), 0, N - 1)
        self.qp.set("lbu", problem.lower, 0, N - 1)
        self.qp.set("ubu", problem.upper, 0, N - 1)
        self.qp.set("Jx", np.eye(nx), 0)
        self.solution = hpipm_ocp_qp_sol(dim)
        self.solution.set("x", np.zeros(nx), 0, N)
        self.solution.set("u", np.zeros(nu), 0, N - 1)
        args = hpipm_ocp_qp_solver_arg(dim, "speed_abs")
        for field, value in {
            "mu0": 1e4, "iter_max": 50, "tol_stat": 1e-3,
            "tol_eq": 1e-5, "tol_ineq": 1e-3, "tol_comp": 1e-3,
            "reg_prim": 1e-12, "warm_start": 1,
        }.items():
            args.set(field, value)
        self.solver = hpipm_ocp_qp_solver(dim, args)

    def step(self, state):
        self.qp.set("lbx", state, 0)
        self.qp.set("ubx", state, 0)
        start = perf_counter_ns()
        self.solver.solve(self.qp, self.solution)
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        status = self.solver.get("status")
        if status != 0:
            raise RuntimeError(f"HPIPM failed: status {status}")
        return self.solution.get("u", 0).reshape(-1), elapsed_ms, self.solver.get("iter")


class DaqpSolver:
    def __init__(self, problem, condensed, K):
        import daqp

        H, G, self.g_x0, _, _, self.lu_x0, self.lower, self.upper = condensed
        self.K = K
        self.nu = problem.B.shape[1]
        self.solver = daqp.Model()
        self.solver.settings = {"primal_tol": 1e-8, "dual_tol": 1e-8,
                                "iter_limit": 4000}
        flag, _ = self.solver.setup(
            H, np.zeros(H.shape[0]), G, self.upper, self.lower,
            np.zeros(G.shape[0], dtype=np.int32),
        )
        if flag < 0:
            raise RuntimeError(f"DAQP setup failed: status {flag}")

    def step(self, state):
        gradient, lower, upper = get_parametric_qp_terms(
            self.g_x0, self.lu_x0, self.lower, self.upper, state,
        )
        self.solver.update(f=gradient, bupper=upper, blower=lower)
        start = perf_counter_ns()
        correction, _, status, info = self.solver.solve()
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        if status != 1:
            raise RuntimeError(f"DAQP failed: status {status}")
        return -self.K @ state + correction[:self.nu], elapsed_ms, info["iterations"]


class QpoasesSolver:
    def __init__(self, problem, condensed, K):
        from pyqpoases import qpoases

        H, G, self.g_x0, _, _, self.lu_x0, self.lower, self.upper = condensed
        self.K = K
        self.solver = qpoases()
        self.solver.init(H, G, np.zeros(H.shape[0]), self.lower, self.upper,
                         problem.N, problem.B.shape[1])

    def step(self, state):
        gradient, lower, upper = get_parametric_qp_terms(
            self.g_x0, self.lu_x0, self.lower, self.upper, state,
        )
        start = perf_counter_ns()
        correction, _, iterations = self.solver.mpcsolve(gradient, lower, upper)
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        return -self.K @ state + correction, elapsed_ms, iterations


def run_benchmark(solver_name="daocp", nreps=1, nsteps=NSTEPS):
    """Run independent closed-loop repetitions, retaining warm starts within each."""
    if nreps < 1 or nsteps < 1:
        raise ValueError("nreps and nsteps must be positive")
    problem = Problem.load()
    if solver_name in ("daqp", "qpoases"):
        K = np.linalg.solve(problem.R + problem.B.T @ problem.P @ problem.B,
                            problem.B.T @ problem.P @ problem.A)
        condensed = condense_input_constrained(
            problem.A, problem.B, problem.Q, problem.R, problem.P,
            problem.lower, problem.upper, problem.N, K,
        )
        adapter = {"daqp": DaqpSolver, "qpoases": QpoasesSolver}[solver_name]
        make_solver = lambda: adapter(problem, condensed, K)
    else:
        adapter = {"daocp": DaocpSolver, "hpipm": HpipmSolver}[solver_name]
        make_solver = lambda: adapter(problem)
    times = np.empty((nreps, nsteps))
    iterations = np.empty((nreps, nsteps), dtype=np.int32)
    states = np.empty((nreps, nsteps + 1, problem.A.shape[0]))
    controls = np.empty((nreps, nsteps, problem.B.shape[1]))
    for rep in range(nreps):
        solver = make_solver()
        states[rep, 0] = problem.x0
        for step in range(nsteps):
            try:
                control, elapsed_ms, iters = solver.step(states[rep, step])
            except RuntimeError as exc:
                raise RuntimeError(f"Repetition {rep + 1}, step {step + 1}: {exc}") from exc
            if not np.all(np.isfinite(control)):
                raise RuntimeError("Solver returned non-finite controls")
            controls[rep, step] = np.clip(control, problem.lower, problem.upper)
            states[rep, step + 1] = (
                problem.A @ states[rep, step] + problem.B @ controls[rep, step]
            )
            times[rep, step] = elapsed_ms
            iterations[rep, step] = iters
        print(f"Repetition {rep + 1}/{nreps}: final state deviation norm "
              f"{np.linalg.norm(states[rep, -1]):.6g}")
    averaged = times.mean(axis=0)
    print(f"Average solve call: {times.mean():.6f} ms")
    print(f"Maximum solve call: {times.max():.6f} ms")
    return dict(solver=np.array(solver_name), nreps=np.array(nreps),
                horizon=np.array(problem.N), nsteps=np.array(nsteps),
                solve_times_ms=times, mean_solve_times_ms=averaged,
                iterations=iterations, states=states, controls=controls)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="save results as an NPZ archive")
    parser.add_argument("--solver", choices=("daocp", "hpipm", "daqp", "qpoases"), default="daocp")
    parser.add_argument("--nreps", type=positive_int, default=1,
                        help="number of independent closed-loop repetitions (default: 1)")
    args = parser.parse_args()
    try:
        results = run_benchmark(args.solver, args.nreps)
    except (ImportError, OSError) as exc:
        parser.exit(1, f"Cannot load {args.solver}: {exc}\n"
                    "Install its Python interface in the active environment and export "
                    "LD_LIBRARY_PATH before starting Python.\n")
    if args.save:
        directory = ROOT / "results"
        directory.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = directory / f"{args.solver}_{timestamp}.npz"
        np.savez_compressed(path, **results)
        print(f"Saved results to {path}")


if __name__ == "__main__":
    main()
