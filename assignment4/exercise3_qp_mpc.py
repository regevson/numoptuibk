import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium.envs.classic_control.pendulum
import numpy as onp
import time
from utils import (
    angle_normalize,
    JaxPendulum,
    make_F_linear,
    linearise_autodiff,
    flatv,
    rowv,
    colv,
    current_step,
)
import qpsolvers
import scipy.sparse as sm
import osqp


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
    # ------------------
    # things to help you get started (not needed afterwards):
    y_test = colv(onp.hstack([onp.zeros(dim_x)] * (horizon + 1) + [onp.zeros(dim_u)] * horizon))
    y_height = dim_x * (horizon + 1) + dim_u * horizon

    P = sm.eye(dim_x * (horizon + 1) + dim_u * horizon)
    q = onp.zeros((dim_x * (horizon + 1) + dim_u * horizon, 1))
    C = sm.coo_matrix((y_height * 2, y_height))
    h = onp.zeros((y_height * 2, 1))
    A = sm.coo_matrix((dim_x * horizon + dim_x + dim_x * int(goal_constraint), y_height))
    b = onp.zeros((dim_x * horizon + dim_x + dim_x * int(goal_constraint), 1))
    # these shapes should work:
    C @ y_test - h
    A @ y_test - b
    y_test.T @ P @ y_test + q.T @ y_test
    # --------------
    # now you need to build all the proper matrices. :-)
    pass # your code here
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
        onp.testing.assert_array_almost_equal(G @ hx[i] + H @ hu[i] + hc[i], hx[i + 1])  # the dynamic constraints are fulfilled.

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
    x_r = jnp.array([0.0, 0.0])  # goal
    # x_r = jnp.array([onp.pi/2, 0.])
    x_initial = jnp.array([-onp.pi, 0.0])
    env = JaxPendulum(initial_state=x_initial, render_mode="human")

    (dim_u,) = env.action_space.low.shape
    (dim_x,) = env.observation_space.low.shape
    xmin = env.observation_space.low * onp.inf
    xmax = env.observation_space.high * onp.inf
    umin = env.action_space.low
    umax = env.action_space.high

    # this are untuned parameters.
    R = onp.eye(1)
    Q = onp.diag(onp.array([1.0, 1.0]))
    Q_final = onp.diag(onp.array([1.0, 1.0]))
    goal_constraint = False
    pass # your code here
    hx, hu = receeding_horizon_MPC(
        env,
        dim_x,
        dim_u,
        x_r,
        R,
        Q,
        30,
        80,
        umin=umin,
        umax=umax,
        xmin=xmin,
        xmax=xmax,
        Q_final=Q_final,
        goal_constraint=goal_constraint,
        render=True,
    )
    plt.subplot(2, 1, 1)
    plt.plot(onp.vstack(hx))
    plt.hlines(x_r[0], *plt.xlim(), linestyles="dashed", color="r", label="goal")
    plt.legend(["Angle", "Velocity", "Angle Goal"])
    plt.hlines(-onp.pi, *plt.xlim(), linestyles="dashed", color="k")
    plt.hlines(onp.pi, *plt.xlim(), linestyles="dashed", color="k")
    plt.subplot(2, 1, 2)
    plt.plot(onp.vstack(hu))
    plt.legend(["Control u"])
    plt.show()


def part3_iteratively_refine_dynamics():
    current_step("Part3, iteratively refine dynamics")

    x_initial = jnp.array([-onp.pi, 0.0])
    env = JaxPendulum(initial_state=x_initial, render_mode="human")
    horizon = 80
    x0, _ = env.reset()
    # ----
    (dim_u,) = env.action_space.low.shape
    (dim_x,) = env.observation_space.low.shape
    xmin = env.observation_space.low
    xmax = env.observation_space.high
    umin = env.action_space.low
    umax = env.action_space.high
    # ---- Parameters to choose and tune.
    goal_constraint = False
    # x_r = jnp.array([0., 0.])  # goal
    x_r = jnp.array([-onp.pi / 2.0, 0.0])  # goal
    R = onp.eye(1) * 0.01
    Q = onp.diag(onp.array([10.0, 1.0]))
    Q_final = onp.diag(onp.array([10.0, 10.0]))

    # ---- Find an initial trajectory over the dynamics you get from linearizing around the starting point x0.
    # - Linearise around the initial point and derive the dynamics parameters for the horizon hG, hH, hc in the same way as for the receeding horizon.
    # - solve for the control sequence and trajectory (see receeding horizon case), this gives you: hx and hu
    #
    pass # your code here
    # For testing purposes rollout the control sequence:
    # render = True
    # x0, _ = env.reset()
    # for u in hu:
    #     x1, _, _, _, _ = env.step(u)
    #     if render:
    #         env.render()
    # It probably doesn't work, because the dynamics are different!

    # --------------------------------------------------- re iterate.
    # repeat n times (try n=10, but experiment with other values if necessary):
    #    - building new hG, hH, hc:
    #        - for i = 0 ... horizon: linearise around hx[i] and use results to build hG, hH, hc
    #    - optimize again for hx and hu.
    #    - compare how much the new hG and the previous hG differ (i.e. onp.linalg.norm(old_hG - new_hG)) (if this doesn't change much, we're done)
    #    - compare how much the new hx and the previous hx differ (i.e. onp.linalg.norm(old_hx - new_hx)) (if this doesn't change much, we're done)
    # when this converges with have a trajectory that adheres to the non-linear dynamics.
    pass # your code here
    # # ---- try running an open loop again:
    # render = True
    # x0, _ = env.reset()
    # for u in hu:
    #     x1, _, _, _, _ = env.step(u)
    #     if render:
    #         env.render()


if __name__ == "__main__":
    part1_setup_the_problem()
    part2_receeding_horizon_mpc()
    part3_iteratively_refine_dynamics()
