import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import qpsolvers
import scipy.sparse as sm
from utils import (
    JaxPendulum,
    colv,
    current_step,
    linearise_autodiff,
)


def to_qps(m):
    if isinstance(m, onp.ndarray):  # dense:
        return m.astype(onp.float64)
    return sm.csc_matrix(m)


def to_full(m):
    if hasattr(m, "toarray"):
        return m.toarray().astype(onp.float64)
    return m.astype(onp.float64)


DEFAULT_SOLVER = "osqp"
to_s = to_qps  # OSQP, otherwise try to_full

# DEFAULT_SOLVER = "cvxopt"
# to_s = to_full  # OSQP, otherwise try to_full


def setup_problem(
    horizon,
    x0,
    x_r,
    dim_x,
    dim_u,
    hG,
    hH,
    hc,
    xmin,
    xmax,
    umin,
    umax,
    Q,
    R,
    Q_final=None,
    goal_constraint=False,
):
    if Q_final is None:
        Q_final = Q

    # Decision variable y = [x_0, x_1, ..., x_N, u_0, u_1, ..., u_{N-1}]
    # y has dimension: dim_x * (horizon + 1) + dim_u * horizon
    N = horizon
    y_height = dim_x * (N + 1) + dim_u * N

    # ==================== Build P and q for cost function ====================
    # Cost: sum_{k=0}^{N-1} [(x_k - x_r)^T Q (x_k - x_r) + u_k^T R u_k] + (x_N - x_r)^T Q_final (x_N - x_r)
    # Rewritten as: 0.5 * y^T P y + q^T y (plus constant terms we ignore)

    # P is block diagonal: [Q, Q, ..., Q, Q_final, R, R, ..., R]
    P_blocks = [Q] * N + [Q_final] + [R] * N
    P = sm.block_diag(P_blocks, format="csc")

    # q comes from the linear term in (x - x_r)^T Q (x - x_r) = x^T Q x - 2 x_r^T Q x + x_r^T Q x_r
    # So q_x = -2 * Q @ x_r for state cost terms (and -2 * Q_final @ x_r for final state)
    q_x = onp.vstack([(-2 * Q @ x_r).reshape(-1, 1)] * N + [(-2 * Q_final @ x_r).reshape(-1, 1)])
    q_u = onp.zeros((dim_u * N, 1))
    q = onp.vstack([q_x, q_u])

    # ==================== Build A and b for equality constraints ====================
    # Constraints:
    # 1. Initial state: x_0 = x0
    # 2. Dynamics: x_{k+1} = G_k x_k + H_k u_k + c_k  =>  -G_k x_k + x_{k+1} - H_k u_k = c_k
    # 3. Optional goal: x_N = x_r

    num_eq_constraints = dim_x + dim_x * N + dim_x * int(goal_constraint)

    # Build A row by row using lists for COO format
    A_rows = []
    A_cols = []
    A_data = []
    b_list = []

    row_offset = 0

    # Constraint 1: x_0 = x0  =>  I @ x_0 = x0
    for i in range(dim_x):
        A_rows.append(row_offset + i)
        A_cols.append(i)
        A_data.append(1.0)
    b_list.append(x0.reshape(-1, 1))
    row_offset += dim_x

    # Constraint 2: Dynamics for k = 0, ..., N-1
    # x_{k+1} - G_k @ x_k - H_k @ u_k = c_k
    for k in range(N):
        G_k = onp.asarray(hG[k])
        H_k = onp.asarray(hH[k])
        c_k = onp.asarray(hc[k]).flatten()

        # -G_k @ x_k: x_k starts at column k * dim_x
        x_k_start = k * dim_x
        for i in range(dim_x):
            for j in range(dim_x):
                if G_k[i, j] != 0:
                    A_rows.append(row_offset + i)
                    A_cols.append(x_k_start + j)
                    A_data.append(-G_k[i, j])

        # I @ x_{k+1}: x_{k+1} starts at column (k+1) * dim_x
        x_kp1_start = (k + 1) * dim_x
        for i in range(dim_x):
            A_rows.append(row_offset + i)
            A_cols.append(x_kp1_start + i)
            A_data.append(1.0)

        # -H_k @ u_k: u_k starts at column dim_x * (N+1) + k * dim_u
        u_k_start = dim_x * (N + 1) + k * dim_u
        for i in range(dim_x):
            for j in range(dim_u):
                if H_k[i, j] != 0:
                    A_rows.append(row_offset + i)
                    A_cols.append(u_k_start + j)
                    A_data.append(-H_k[i, j])

        b_list.append(c_k.reshape(-1, 1))
        row_offset += dim_x

    # Constraint 3: Goal constraint x_N = x_r (optional)
    if goal_constraint:
        x_N_start = N * dim_x
        for i in range(dim_x):
            A_rows.append(row_offset + i)
            A_cols.append(x_N_start + i)
            A_data.append(1.0)
        b_list.append(x_r.reshape(-1, 1))
        row_offset += dim_x

    A = sm.coo_matrix((A_data, (A_rows, A_cols)), shape=(num_eq_constraints, y_height))
    b = onp.vstack(b_list)

    # ==================== Build C and h for inequality constraints ====================
    # Box constraints: xmin <= x_k <= xmax, umin <= u_k <= umax
    # Written as: C @ y <= h
    # For each variable v with vmin <= v <= vmax:
    #   v <= vmax  =>  v <= vmax
    #   -v <= -vmin  =>  v >= vmin

    # Number of inequality constraints: 2 * (dim_x * (N+1) + dim_u * N)
    num_ineq = 2 * y_height

    C_rows = []
    C_cols = []
    C_data = []
    h_list = []

    row_offset = 0

    # State constraints for x_0, ..., x_N
    for k in range(N + 1):
        x_k_start = k * dim_x
        for i in range(dim_x):
            # x_k[i] <= xmax[i]
            C_rows.append(row_offset)
            C_cols.append(x_k_start + i)
            C_data.append(1.0)
            h_list.append(xmax[i])
            row_offset += 1

            # -x_k[i] <= -xmin[i]
            C_rows.append(row_offset)
            C_cols.append(x_k_start + i)
            C_data.append(-1.0)
            h_list.append(-xmin[i])
            row_offset += 1

    # Control constraints for u_0, ..., u_{N-1}
    for k in range(N):
        u_k_start = dim_x * (N + 1) + k * dim_u
        for i in range(dim_u):
            # u_k[i] <= umax[i]
            C_rows.append(row_offset)
            C_cols.append(u_k_start + i)
            C_data.append(1.0)
            h_list.append(umax[i])
            row_offset += 1

            # -u_k[i] <= -umin[i]
            C_rows.append(row_offset)
            C_cols.append(u_k_start + i)
            C_data.append(-1.0)
            h_list.append(-umin[i])
            row_offset += 1

    C = sm.coo_matrix((C_data, (C_rows, C_cols)), shape=(num_ineq, y_height))
    h = onp.array(h_list).reshape(-1, 1)

    return P, q, C, h, A, b


def receeding_horizon_MPC(
    env,
    dim_x,
    dim_u,
    x_r,
    R,
    Q,
    horizon,
    steps,
    umin=None,
    umax=None,
    xmin=None,
    xmax=None,
    Q_final=None,
    goal_constraint=False,
    render=True,
):
    x0, _ = env.reset()
    F_step = env.F_step
    if Q_final is None:
        Q_final = Q
    trajectory = [x0]
    control_sequence = []
    res = None
    for i in range(steps):
        G, H, c = linearise_autodiff(F_step, x0)
        hG = [G] * horizon
        hH = [H] * horizon
        hc = [c] * horizon
        P, q, C, h, A, b = setup_problem(
            horizon,
            x0,
            x_r,
            dim_x,
            dim_u,
            hG,
            hH,
            hc,
            xmin,
            xmax,
            umin,
            umax,
            Q,
            R,
            Q_final=Q_final,
            goal_constraint=goal_constraint,
        )
        res = qpsolvers.solve_qp(
            to_s(P),
            to_s(q).squeeze(),
            G=to_s(C),
            h=to_s(h).squeeze(),
            A=to_s(A),
            b=to_s(b).squeeze(),
            solver=DEFAULT_SOLVER,
            initvals=res,
        )
        assert res is not None
        y = res
        hx = y[: (horizon + 1) * dim_x].reshape(horizon + 1, dim_x)
        hu = y[-horizon * dim_u :].reshape(horizon, dim_u)
        u = hu[0]
        x1, _, _, _, _ = env.step(u)
        if render:
            env.render()
        trajectory.append(x1)
        control_sequence.append(u)
        x0 = x1  # next step.
    return trajectory, control_sequence


def part1_setup_the_problem():
    current_step("Part1, setup the problem")
    ### Part 1 -- setting up the problem. Here is test code to help you test your solution.
    # you need to build the setup_problem function. This is to test your code.
    # Failure of this code implies your code is wrong.
    x_initial = jnp.array([-onp.pi, 0.0])
    env = JaxPendulum(initial_state=x_initial, render_mode="human")

    (dim_u,) = env.action_space.low.shape
    (dim_x,) = env.observation_space.low.shape
    xmin = env.observation_space.low * onp.inf  # we don't care about state limits
    xmax = env.observation_space.high * onp.inf
    umin = env.action_space.low  # we only care about action limits
    umax = env.action_space.high

    horizon = 3
    x0, _ = env.reset()
    x_r = x0  # staying at the initial position should definitely be possible (in our case).
    R = onp.eye(dim_u)
    Q = onp.eye(dim_x)
    Q_final = Q

    G, H, c = linearise_autodiff(env.F_step, x0)
    hG = [G] * horizon
    hH = [H] * horizon
    hc = [c] * horizon

    P, q, C, h, A, b = setup_problem(
        horizon,
        x0,
        x_r,
        dim_x,
        dim_u,
        hG,
        hH,
        hc,
        xmin,
        xmax,
        umin,
        umax,
        Q,
        R,
        Q_final=Q_final,
        goal_constraint=False,
    )
    # Test the shapes:
    y_test = colv(onp.hstack([onp.zeros(dim_x)] * (horizon + 1) + [onp.zeros(dim_u)] * horizon))
    C @ y_test - h
    A @ y_test - b
    y_test.T @ P @ y_test + q.T @ y_test

    # Try to solve it:
    res = qpsolvers.solve_qp(
        to_s(P),
        to_s(q).squeeze(),
        G=to_s(C),
        h=to_s(h).squeeze(),
        A=to_s(A),
        b=to_s(b).squeeze(),
        solver=DEFAULT_SOLVER,
    )
    assert res is not None  # your problem could not be solved. This means there is a problem.

    # Test the shapes with goal constraint:
    P, q, C, h, A, b = setup_problem(
        horizon,
        x0,
        x_r,
        dim_x,
        dim_u,
        hG,
        hH,
        hc,
        xmin,
        xmax,
        umin,
        umax,
        Q,
        R,
        Q_final=Q_final,
        goal_constraint=True,
    )
    # Test the shapes:
    y_test = colv(onp.hstack([onp.zeros(dim_x)] * (horizon + 1) + [onp.zeros(dim_u)] * horizon))
    C @ y_test - h
    A @ y_test - b
    y_test.T @ P @ y_test + q.T @ y_test

    # Try to solve it:
    res = qpsolvers.solve_qp(
        to_s(P),
        to_s(q).squeeze(),
        G=to_s(C),
        h=to_s(h).squeeze(),
        A=to_s(A),
        b=to_s(b).squeeze(),
        solver=DEFAULT_SOLVER,
    )
    assert res is not None  # your problem could not be solved. This means there is a problem.

    # now try to plan a trajectory...
    horizon = 80
    hG = [G] * horizon
    hH = [H] * horizon
    hc = [c] * horizon
    x_r = jnp.array([0.0, 0.0])

    # Test the shapes with goal constraint:
    P, q, C, h, A, b = setup_problem(
        horizon,
        x0,
        x_r,
        dim_x,
        dim_u,
        hG,
        hH,
        hc,
        xmin,
        xmax,
        umin,
        umax,
        Q,
        R,
        Q_final=Q_final,
        goal_constraint=True,
    )
    res = qpsolvers.solve_qp(
        to_s(P),
        to_s(q).squeeze(),
        G=to_s(C),
        h=to_s(h).squeeze(),
        A=to_s(A),
        b=to_s(b).squeeze(),
        solver=DEFAULT_SOLVER,
    )
    assert res is not None  # your problem could not be solved. This means there is a problem.
    hx = res[: (horizon + 1) * dim_x].reshape(horizon + 1, dim_x)
    hu = res[-horizon * dim_u :].reshape(horizon, dim_u)

    for i in range(horizon):
        onp.testing.assert_array_almost_equal(
            G @ hx[i] + H @ hu[i] + hc[i], hx[i + 1]
        )  # the dynamic constraints are fulfilled.

    onp.testing.assert_array_almost_equal(hx[0], x0)  # initial constraint
    onp.testing.assert_array_almost_equal(hx[-1], x_r)  # goal constraint

    # check whether the boundary constraints are fulfilled.
    eps = 1e-4
    assert onp.all(umin - eps <= hu)
    assert onp.all(hu <= umax + eps)
    assert onp.all(xmin - eps <= hx)
    assert onp.all(hx <= xmax + eps)

    # now let's test whether the cost is decreasing...
    cost0 = (hx[0] - x_r).T @ Q @ (hx[0] - x_r) + (hu[0].T @ R @ hu[0])
    costl = (hx[-1] - x_r).T @ Q @ (hx[-1] - x_r) + (hu[-1].T @ R @ hu[-1])
    assert cost0 >= costl  # the cost should be decreasing as we move closer to the goal.


def part2_receeding_horizon_mpc():
    current_step("Part2, receeding horizon")

    ### Part 2 -- receeding horizon
    # Analyse the code of the receeding_horizon_MPC function! (be able to explain what it does)
    # Try the 3 different goal points and plot the results.
    # Tune R, Q, Q_final!

    # Three goal positions to test
    goal_positions = [
        jnp.array([0.0, 0.0]),  # Goal 1: upright position
        jnp.array([onp.pi / 2, 0.0]),  # Goal 2: horizontal right
        jnp.array([-onp.pi / 2, 0.0]),  # Goal 3: horizontal left
    ]
    goal_names = ["[0, 0] (upright)", "[π/2, 0] (horizontal right)", "[-π/2, 0] (horizontal left)"]

    x_initial = jnp.array([-onp.pi, 0.0])

    for goal_idx, x_r in enumerate(goal_positions):
        print(f"\n--- Testing goal {goal_idx + 1}: {goal_names[goal_idx]} ---")

        env = JaxPendulum(initial_state=x_initial, render_mode="human")

        (dim_u,) = env.action_space.low.shape
        (dim_x,) = env.observation_space.low.shape
        xmin = env.observation_space.low * onp.inf  # No state constraints
        xmax = env.observation_space.high * onp.inf
        umin = env.action_space.low
        umax = env.action_space.high

        # Tuned parameters:
        # - R = 0.1: Low penalty on control effort allows stronger actions
        # - Q: High weight (10) on angle to prioritize reaching goal angle,
        #      lower weight (0.1) on velocity to allow faster movement
        # - Q_final: Higher weights to ensure we stop at the goal
        R = onp.eye(1) * 0.1
        Q = onp.diag(onp.array([10.0, 0.1]))
        Q_final = onp.diag(onp.array([100.0, 10.0]))
        goal_constraint = False

        hx, hu = receeding_horizon_MPC(
            env,
            dim_x,
            dim_u,
            x_r,
            R,
            Q,
            30,  # horizon
            80,  # steps
            umin=umin,
            umax=umax,
            xmin=xmin,
            xmax=xmax,
            Q_final=Q_final,
            goal_constraint=goal_constraint,
            render=True,
        )

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.suptitle(f"Goal: {goal_names[goal_idx]}")

        plt.subplot(2, 1, 1)
        plt.plot(onp.vstack(hx))
        plt.hlines(x_r[0], *plt.xlim(), linestyles="dashed", color="r", label="Angle Goal")
        plt.hlines(x_r[1], *plt.xlim(), linestyles="dotted", color="g", label="Velocity Goal")
        plt.ylabel("State")
        plt.legend(["Angle", "Velocity", "Angle Goal", "Velocity Goal"])
        plt.hlines(-onp.pi, *plt.xlim(), linestyles="dashed", color="k", alpha=0.3)
        plt.hlines(onp.pi, *plt.xlim(), linestyles="dashed", color="k", alpha=0.3)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(onp.vstack(hu))
        plt.hlines(umin[0], *plt.xlim(), linestyles="dashed", color="r", alpha=0.5)
        plt.hlines(umax[0], *plt.xlim(), linestyles="dashed", color="r", alpha=0.5)
        plt.ylabel("Control Input")
        plt.xlabel("Time Step")
        plt.legend(["Control u", "u_min", "u_max"])
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"part2_goal_{goal_idx + 1}.png", dpi=150)
        plt.show()


def part3_iteratively_refine_dynamics():
    import time

    current_step("Part3, iteratively refine dynamics")

    x_initial = jnp.array([-onp.pi, 0.0])
    env = JaxPendulum(initial_state=x_initial, render_mode="human")
    horizon = 80
    x0, _ = env.reset()
    # ----
    (dim_u,) = env.action_space.low.shape
    (dim_x,) = env.observation_space.low.shape
    xmin = env.observation_space.low * onp.inf  # No state constraints as per logic usually, but let's see.
    # Actually assignment 3.1 code had xmin/xmax. Let's keep them infinite like in part 2 unless specified.
    # The provided stub code has: xmin = env.observation_space.low (which is finite).
    # But usually for pendulum swing up we might want to relax them or keep them.
    # The stub code had: xmin = env.observation_space.low
    # Let's trust the stub code lines 376-377.
    xmin = env.observation_space.low
    xmax = env.observation_space.high
    umin = env.action_space.low
    umax = env.action_space.high

    # ---- Parameters to choose and tune.
    goal_constraint = True  # Requirement: "goal constraint ... should, if set to true, add a constraint"
    # Instructions don't explicitly say to turn it on/off for Part 3, but "optimize again for hx and hu" implies we want to reach the goal.

    x_r = jnp.array([-onp.pi / 2.0, 0.0])  # goal
    R = onp.eye(1) * 0.01
    Q = onp.diag(onp.array([10.0, 1.0]))
    Q_final = onp.diag(onp.array([10.0, 10.0]))

    # ---- Find an initial trajectory over the dynamics you get from linearizing around the starting point x0.
    print("Initial Solve: Linearizing around x0")
    # Linearize around (x0, 0) for all steps
    G0, H0, c0 = linearise_autodiff(env.F_step, x0, jnp.zeros(dim_u))
    hG = [G0] * horizon
    hH = [H0] * horizon
    hc = [c0] * horizon

    P, q, C, h, A, b = setup_problem(
        horizon,
        x0,
        x_r,
        dim_x,
        dim_u,
        hG,
        hH,
        hc,
        xmin,
        xmax,
        umin,
        umax,
        Q,
        R,
        Q_final=Q_final,
        goal_constraint=goal_constraint,
    )

    res = qpsolvers.solve_qp(
        to_s(P),
        to_s(q).squeeze(),
        G=to_s(C),
        h=to_s(h).squeeze(),
        A=to_s(A),
        b=to_s(b).squeeze(),
        solver=DEFAULT_SOLVER,
    )

    if res is None:
        print("Initial QP failed to solve!")
        return

    hx = res[: (horizon + 1) * dim_x].reshape(horizon + 1, dim_x)
    hu = res[-horizon * dim_u :].reshape(horizon, dim_u)

    print(f"Initial trajectory cost: {(hx[-1] - x_r).T @ Q_final @ (hx[-1] - x_r)}")

    # Plot initial guess
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(onp.vstack(hx))
    plt.title("Initial Linear Trajectory")

    # Render initial guess (Open Loop)
    print("Rolling out initial control sequence...")
    # env.reset()
    # env.state = x0
    # for u in hu:
    #     env.step(u)
    #     env.render()
    #     time.sleep(0.01)

    # --------------------------------------------------- re iterate.
    num_iterations = 10

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}")

        # 1. Rollout with nonlinear dynamics to get linearization points
        #    xi = Fstep(hx[i-1], hu[i-1]) ... effectively rolling using previous controls
        #    We linearize around these (x_traj[i], hu[i]) points.

        x_linearize = [x0]
        curr_x = x0

        new_hG = []
        new_hH = []
        new_hc = []

        # We need to linearize at k=0...horizon-1 to get constraints for x_{k+1}
        for k in range(horizon):
            u_k = hu[k]

            # Linearize around current predicted state and control
            G_k, H_k, c_k = linearise_autodiff(env.F_step, curr_x, u_k)
            new_hG.append(G_k)
            new_hH.append(H_k)
            new_hc.append(c_k)

            # Propagate to get next linearization point
            curr_x = env.F_step(curr_x, u_k).reshape(dim_x)
            x_linearize.append(curr_x)

        x_linearize = onp.array(x_linearize)

        # 2. Optimize again
        P, q, C, h, A, b = setup_problem(
            horizon,
            x0,
            x_r,
            dim_x,
            dim_u,
            new_hG,
            new_hH,
            new_hc,
            xmin,
            xmax,
            umin,
            umax,
            Q,
            R,
            Q_final=Q_final,
            goal_constraint=goal_constraint,
        )

        # Warm start? osqp supports it but qpsolvers interface might not fully for all solvers
        # We pass initvals=res
        new_res = qpsolvers.solve_qp(
            to_s(P),
            to_s(q).squeeze(),
            G=to_s(C),
            h=to_s(h).squeeze(),
            A=to_s(A),
            b=to_s(b).squeeze(),
            solver=DEFAULT_SOLVER,
            initvals=res,
        )

        if new_res is None:
            print(f"QP failed at iteration {iteration}!")
            break

        new_hx = new_res[: (horizon + 1) * dim_x].reshape(horizon + 1, dim_x)
        new_hu = new_res[-horizon * dim_u :].reshape(horizon, dim_u)

        # 3. Compare changes
        diff_hx = onp.linalg.norm(hx - new_hx)
        diff_hu = onp.linalg.norm(hu - new_hu)

        print(f"Diff hx: {diff_hx:.5f}, Diff hu: {diff_hu:.5f}")

        hx = new_hx
        hu = new_hu
        res = new_res

        if diff_hx < 1e-2 and diff_hu < 1e-2:
            print("Converged!")
            break

    plt.subplot(1, 2, 2)
    plt.plot(onp.vstack(hx))
    plt.title("Refined Trajectory")
    plt.hlines(x_r[0], 0, horizon, colors="r", linestyles="dashed", label="Goal")
    plt.legend()
    plt.savefig("part3_refinement.png")
    plt.show()

    # ---- try running an open loop again:
    print("\nVisualizing final open-loop rollout...")
    render = True
    x0, _ = env.reset()
    rollout_traj = [x0]

    for u in hu:
        x1, _, _, _, _ = env.step(u)
        rollout_traj.append(x1)
        if render:
            env.render()
            time.sleep(0.02)

    rollout_traj = onp.vstack(rollout_traj)
    final_error = onp.linalg.norm(rollout_traj[-1] - x_r)
    print(f"Final position: {rollout_traj[-1]}")
    print(f"Target: {x_r}")
    print(f"Euclidean Error: {final_error}")

    if final_error < 0.1:
        print("SUCCESS: Controller reached the goal!")
    else:
        print("WARNING: Did not perfectly reach the goal in open loop (expected due to small model mismatch).")


if __name__ == "__main__":
    part1_setup_the_problem()
    part2_receeding_horizon_mpc()
    part3_iteratively_refine_dynamics()
