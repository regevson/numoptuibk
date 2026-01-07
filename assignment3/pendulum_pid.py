import gymnasium as gym
import gymnasium.envs.classic_control.pendulum
import numpy as np
import matplotlib.pyplot as plt
import functools
import time
import plotille
## In this exercise you are given the pendulum environment.
# This is based on gym.make("Pendulum-v1")
# i.e. gym.envs.classic_control.pendulum.PendulumEnv Differences: we
# directly take the state as observations (th, thdot), instead of
# (sin(th), cos(th), thdot), and we add some friction in the pendulum,
# because that makes our life easier.
# Also this one is not under-actuated (i.e. the motor has enough power)


class Pendulum(gym.envs.classic_control.pendulum.PendulumEnv):
    def __init__(self, render_mode=None):
        super(Pendulum, self).__init__(render_mode=render_mode)
        self.max_torque = 30.0
        self.action_space = gym.spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.goal = 0.0  # top position
        self.dt = 0.05  # 0.01
        self.state = None  # need to reset first
        self.last_u = None

    def reset(self, seed=None, options=None, angle=None):
        super().reset(seed=seed)
        if options is not None:
             angle = options.get("angle", angle)

        if angle is None:
            angle = np.deg2rad(-40)
        self.state = np.array([angle, 0.0])
        self.last_u = None
        return self.state, {}

    def step(self, u):
        _obs, reward, terminated, truncated, info = super(Pendulum, self).step(u)
        th, thdot = self.state
        delta_thdot = -0.01 * thdot  # friction
        thdot = thdot + delta_thdot
        th = th + delta_thdot * self.dt
        th = (th + np.pi) % (2 * np.pi) - np.pi
        self.state = np.array([th, thdot])
        return self.state, reward, terminated, truncated, info


gym.envs.register(id="StatePendulum-v1", entry_point=Pendulum, max_episode_steps=200)


# This function performs a rollout using your controller function
#
# ufunc(Array[2]) -> Array[1]
#
# that maps the state to a control input.
def rollout(
    env: gym.Env,
    ufunc,
    err_reset=lambda state: np.array([0.0, 0.0, 0.0]),
    err_step=lambda err_state, state: err_state,
    render=True,
    print_states=False,
    reset_angle=None,
):
    """
    """
    state, _info = env.reset(options={"angle": reset_angle})
    err_state = err_reset(state)
    done = False
    states = [state]
    err_states = [err_state]
    cost = 0.0
    while not done:
        u = ufunc(err_state)
        state, reward, terminated, truncated, _info = env.step(u)
        done = terminated or truncated
        err_state = err_step(err_state, state)
        if render:
            env.render()
        if print_states:
            print(state, err_state, u)
        states.append(state)
        err_states.append(err_state)
        cost -= reward
    time.sleep(2)
    return np.vstack(states), np.vstack(err_states), cost


# example u_func:
def go_right(err_state):
    del err_state
    return np.array([2.0])


if __name__ == "__main__":
    np.set_printoptions(formatter=dict(float=lambda number: "%9.3f" % (number,)))
    render_env = gym.make("StatePendulum-v1", render_mode="human")
    render_env.metadata["render_fps"] = 200 # fps
    env = gym.make("StatePendulum-v1")
    env.reset()

    # Now you have to implement the necessary components for the PID
    # controller, that is you need the error signal, the integration
    # over the error signal and the derivative of the error signal.
    # And put that in a vector [int_error, error, dot_error]
    # where "error" is the difference between the current position of
    # the pendulum and the goal (env.goal)

    # Task 1.) (1 Point) Implement
    #
    # err_reset(state_0) -> error_state_0
    #
    # and
    #
    # err_step(error_state_k, state_k) -> error_state_{k+1}

    # in the rollout function we will use err_reset together with
    # env.reset() to initialize your error-signal tracker.  Then
    # env.step(action) is used to advance the environment state, and
    # your err_step function is used to advance the error_{k+1} based
    # on the last error_{k} and state_{k+1}.  Different to our
    # previous discrete time systems, the error signals current state
    # can be directly calculated:
    #
    # error_k+1 = theta_{k+1} - goal
    #
    # While   int_error is the integration of the error
    # and     dot_error is the derivative of the error
    #
    # So you have to do both a numerical integration (for int_error)
    # as well as a numerical derivative (for dot_error).

    def err_reset(state):
        th, thdot = state
        return np.array([0.0, th - env.unwrapped.goal, 0.0])

    def err_step(err_state, state):
        (th, _thdot) = state
        err_i_k, err_k, err_dot_k = err_state # prev error-vec
        
        current_error = th - env.unwrapped.goal
        dot_error = (current_error - err_k) / env.unwrapped.dt # change in error
        int_error = err_i_k + current_error * env.unwrapped.dt
        
        err_state_kp1 = np.array([int_error, current_error, dot_error])
        return err_state_kp1

    # you can try your solution with
    # rollout(render_env, go_right, err_reset=err_reset, err_step=err_step, render=True, print_states=True)

    # Task 2.) (1 Point) build a configurable controller function with configurable gains kI, kP, kD:
    # return a function err_state -> u
    def make_pid_controller(kP, kI, kD):
        def controller(err_state):
            err_i, err_p, err_d = err_state
            u = kP * err_p + kI * err_i + kD * err_d
            return np.array([u])
        return controller

    # Task 3.) (1 Point) Tune your controller from the previous step using the Ziegler-Nichols method!
    
    # my configuration-try, that doesn't really work...
    Tu = 0.4
    Ku = 20.0
    kP = Ku
    kI = 0.0
    kD = 0.0

    kP = 0.6 * Ku
    kI = 2 * kP / Tu
    kD = kP * Tu / 8
    
    pid_ufunc = make_pid_controller(kP, kI, kD)
    ### Plot the result!
    # use render_env instead of env to visualize!
    states, err_states, cost = rollout(render_env, pid_ufunc, err_reset=err_reset, err_step=err_step, render=True, print_states=True, reset_angle=np.deg2rad(-40.)) # in Task 4, try with -180

    t_x = np.arange(len(states)) * env.unwrapped.dt
    fig = plotille.Figure()
    fig.width = 80
    fig.height = 30
    fig.set_y_limits(min_=-np.pi, max_=np.pi)
    fig.color_mode = 'byte'
    fig.plot(t_x, states[:, 0])
    print(fig.show())

    plt.plot(t_x, err_states[:, 0], label="int_error")
    plt.plot(t_x, err_states[:, 1], label="error")
    plt.plot(t_x, err_states[:, 2], label="dot_error")
    plt.hlines(env.unwrapped.goal, 0, len(err_states) *env.unwrapped.dt, color="r", label="goal")
    plt.title(f"cost: {cost})")
    plt.legend()
    plt.show()


    # Task 4.) (1 Point) Further optimize the kp, ki, kd parameters, but for a statring state (reset_angle of -180 degrees!)
    # In the previous task you used Ziegler-Nichols to tune the PID and the plot showed the cost in the title.
    # Does the Ziegler-Nichols method work equally well?
    # How small can you get the cost by tuning kp, ki, kd?
    TODO