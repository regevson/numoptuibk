import gymnasium as gym
import numpy as np
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt
import time
import tqdm
import pygame

def current_step(title):
    print("--------------------------------------------------------------------------------")
    print(title)
    pygame.display.set_caption(title)
    print("\n")


### Utility for plotting heatmaps
class GridEval:
    def __init__(self, grid_evaluate, x1_min, x1_max, x2_min, x2_max, x1steps=10, x2steps=10):
        k1 = np.linspace(x1_min, x1_max, x1steps)
        k2 = np.linspace(x2_min, x2_max, x2steps)
        grid_points = cartesian([k1, k2])
        result = np.vstack([grid_evaluate(gp) for gp in tqdm.tqdm(grid_points)])
        grid_result = result.reshape((k1.shape[0], k2.shape[0])).T
        self.k1 = k1
        self.k2 = k2
        self.grid_result = grid_result

    def plot(self, close=True, title=None):
        k1 = self.k1
        k2 = self.k2
        if close:
            plt.close()
        plt.imshow(self.grid_result, extent=[k1[0], k1[-1], k2[0], k2[-1]], aspect="auto", origin="lower")
        plt.xlabel("k1")
        plt.ylabel("k2")
        plt.colorbar()
        if title:
            plt.title(title)
        plt.show()


# we can define a do-nothing controller:
def U_go_right(x):
    return [1]


def render(env, U_func, n=1):
    for i in range(n):
        episode_return = 0
        state, _info = env.reset()
        done = False
        while not done:
            state, reward, terminated, truncated, info = env.step(U_func(state))
            done = terminated or truncated
            env.render()
            # time.sleep(0.01)


if __name__ == "__main__":
    # In this exercise we take a first look at defining controllers (policies).
    # A controller/policy provides us with a control input given a state: u = f(x)

    # we first instantantiate the environment
    render_env = gym.make("MountainCarContinuous-v0", render_mode="human")
    render_env.metadata["render_fps"] = 200 # fps
    env = gym.make("MountainCarContinuous-v0")

    # above there is a pre-defined controller that simply tries to
    # drive up the mountain.  we can visualise the struggle:

    current_step("0.) Testing environment by going full force to the right")
    render(render_env, U_go_right)

    # 1.) (1 Point) Define the controller as a linear function of the state:
    #     u=Kx   K is a 1x2 matrix.
    # Create a new function that takes K as input, and returns a new control function.
    # K is called the control-gain matrix.
    # try some values for the control gain and render.

    def Kfunc(K):
        return lambda x: np.atleast_1d(np.dot(K, x)) # the k-func returns a u-func (it just puts in a constant K into the u-func)

    # 2.) (1 Point) write an evaluation function that uses a given control
    current_step("2.) your evaluation function")
 
    # function, resets the environment and does n (=5 as default)
    # rollouts with your controller. accumulating the reward
    # (=-cost).:

    # define an evaluate function
    # def evaluate(env, U_func, n=5):
    # - that loops n-times
    #  - sets the initial state
    #  - loop until the done flag is true
    #   - steps according to u(state)
    #   - accumulates the reward
    #  - take the sum_of_the_reward
    # - take the mean of the [sum_of_the_reward, ...], and return that.
    def evaluate(env, U_func, n=5):
        total_returns = 0
        for _ in range(n):
            state, _ = env.reset()
            done = False
            episode_return = 0
            while not done:
                state, reward, terminated, truncated, _ = env.step(U_func(state))
                episode_return += reward
                done = terminated or truncated
            total_returns += episode_return
        return total_returns / n

    # 3.) (1 Point) Since the gain matrix is 1x2 we can visualize the
    # quality (== what the evaluate function gives us) for different
    # gain parameters as a plot.  the function should take k1, k2 as
    # input and return how good (evaluate) that controller is.  Use
    # that plot to select k1, k2 to get a good controller.
    current_step("3.) grid-evaluate controller parameters")

    def grid_eval(data): # what K should we choose for the u-func?
        k1, k2 = data
        K = np.array([k1, k2])
        return evaluate(env, Kfunc(K))

    # and with this code we can plot a heatmap.
    current_step("3.) plot evaluation of controller parameters")
    grid = GridEval(grid_eval, -10, 10, -10, 10)
    grid.plot(title="plot evaluation of controller parameters")

    # 4.) (1 Point) Phase-space plots: Since the state space of the
    # mountain car is two dimensional we can plot trajectories that
    # the car takes as trajectories through the phase space
    #
    #(i.e.: x = position, y = velocity)
    #
    # define a rollout function
    #
    # def rollout(env, U_func) -> array of shape [time,2]
    #
    # select 2 different K matrices based on the gripmap evaluation
    # above and plot the resulting trajectories in the phase-space.
    current_step("4.) plot phase-space trajectory")

    def rollout(env, U_func): # collect the states of a rollout -> this array is the trajectory
        trajectory = []
        state, _ = env.reset()
        trajectory.append(state)
        done = False
        while not done:
            state, _, terminated, truncated, _ = env.step(U_func(state))
            trajectory.append(state)
            done = terminated or truncated
        return np.array(trajectory)

    # Select 2 different K matrices
    K1 = np.array([0, 10])
    K2 = np.array([0, -10])

    traj1 = rollout(env, Kfunc(K1))
    traj2 = rollout(env, Kfunc(K2))

    plt.close()
    plt.plot(traj1[:, 0], traj1[:, 1], label=f"K={K1}")
    plt.plot(traj2[:, 0], traj2[:, 1], label=f"K={K2}")
    plt.legend()
    plt.title("Plot phase-space trajectory")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.show()

    current_step("5.) show the controller in action")
    render(render_env, Kfunc(K1))
