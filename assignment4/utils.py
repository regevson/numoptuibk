import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium.envs.classic_control.pendulum
import numpy as onp
import time
import pygame

def current_step(title):
    print(
        "--------------------------------------------------------------------------------"
    )
    print(title)
    pygame.display.set_caption(title)
    print("\n")


def assert_flat(vector):
    # raise error if the vector is NOT flat.
    assert len(vector.shape) == 1


def flatv(vector):
    # convert vector to flat vector of shap [n]
    v = vector.squeeze()
    assert_flat(v), f"you didn't pass a vector: {vector.shape}"
    return v


def rowv(vector):
    # convert vector to a rowvector of shape [1, n]
    return flatv(vector)[None, :]


def colv(vector):
    # convert vector to a column vector of shape [n, 1]
    return flatv(vector)[:, None]


def angle_normalize(x):
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


class JaxPendulum(gym.envs.classic_control.pendulum.PendulumEnv):
    def __init__(
        self,
        friction=0.0,
        initial_state=jnp.array([jnp.pi, 0.0]),
        max_torque=7.0,
        max_speed=8.0,
        render_mode=None,
    ):
        super().__init__(render_mode=render_mode)
        self.max_torque = max_torque
        self.max_speed = max_speed
        self.action_space = gym.spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=jnp.float32
        )
        self.goal = 0.0  # top position
        self.dt = 0.05  # 0.01
        self.state = None  # need to reset first
        self.last_u = None
        self.friction = friction
        obs_limit = jnp.array([jnp.pi, self.max_speed])
        self.observation_space = gym.spaces.Box(
            low=-onp.asarray(obs_limit), high=onp.asarray(obs_limit)
        )
        self.initial_state = initial_state
        # prepare the stepper closure.
        self.F_step = self.make_step_function()

    def reset(self):
        self.state = self.initial_state
        self.last_u = None
        return self.state, {}

    def make_step_function(self):
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        max_speed = self.max_speed
        friction = self.friction

        def F_step(x_k, u_k):
            th, thdot = jnp.split(x_k, 2, axis=-1)
            th = jnp.reshape(th, (-1,))
            thdot = jnp.reshape(thdot, (-1,))
            # actual physics for acceleration
            newthdot = (
                thdot + (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l**2) * u_k) * dt
            )
            # apply friction
            newthdot = (1 - friction) * newthdot
            # integrate
            newth = th + newthdot * dt
            # angle normalization
            newth = (newth + jnp.pi) % (2 * jnp.pi) - jnp.pi
            # clipping speed
            newthdot = jnp.clip(newthdot, -max_speed, max_speed)
            # ---
            x_kp1 = jnp.hstack([newth[:, None], newthdot[:, None]])
            return x_kp1.reshape(x_k.shape)

        return F_step

    def step(self, u):
        th, thdot = self.state  # th := theta

        # g = self.g
        # m = self.m
        # l = self.l
        # dt = self.dt

        u = jnp.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        # # cost is based on the previous step.
        # costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
        costs = 0  # no information

        self.state = self.F_step(self.state, u)
        return self.state, -costs, False, False, {}


def linearise_autodiff(f_step, x_prime=jnp.zeros((2,)), u_prime=jnp.zeros((1,))):
    G = jax.jacobian(f_step, 0)(x_prime, u_prime)
    H = jax.jacobian(f_step, 1)(x_prime, u_prime)
    c = f_step(x_prime, u_prime) - (G @ x_prime + H @ u_prime)
    return G, H, c


def make_F_linear(G, H, c):
    def linear_F_step(x, u):
        return G @ x + H @ u.reshape(-1) + c

    return linear_F_step
