import gym
import copy
import torch
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

# Demonstration
env = gym.envs.make("CartPole-v1")


def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.
    return torch.from_numpy(screen)


def ploter(values, title=''):
    # Update the window after each episode
    clear_output(wait=True)

    # Define the figure
    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    axes[0].plot(values, label='score per run')
    axes[0].axhline(195, c='green', ls='--', label='goal')
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Reward')
    x = range(len(values))
    axes[0].legend()
    # Calculate the trend
    z = np.polyfit(x, values, 1)
    p = np.poly1d(z)
    axes[0].plot(x, p(x), "--", label='trend')

    # Plot the histogram of results
    axes[1].hist(values[-50:])
    axes[1].axvline(195, c='green', label='goal')
    axes[1].set_xlabel('Scores per Last 50 Episodes')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    plt.show()


def basic_search(env, episodes, title=''):
    final = []
    for episode in range(episodes):
        state = env.reset()
        finish = False
        reward_calc = 0
        while not finish:
            action = env.action_space.sample()  # Sample random actions
            next_state, reward, finish, _ = env.step(action)  # Take action and extract results
            reward_calc += reward  # Update reward
            if finish:
                break
        # Add to the final reward
        final.append(reward_calc)
        # plot the results
        if episode == (episodes - 1):
            ploter(final, title)
    return final


def Q_Learning(env, model, episodes, gamma, epsilon, eps_decay=0.99, replay=False, replay_size=20,
               title='DQN', double=False, n_update=10):
    final = []
    memory = []
    for episode in range(episodes):
        if double:
            # Update target network every n_update steps
            if episode % n_update == 0:
                model.target_update()

        # Reset state
        state = env.reset()
        finish = False
        reward_calc = 0

        while not finish:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                Q_values = model.predict(state)
                action = torch.argmax(Q_values).item()

            # Take action and add reward to total
            next_state, reward, finish, _ = env.step(action)

            # Update total and memory
            reward_calc += reward
            memory.append((state, action, next_state, reward, finish))
            Q_values = model.predict(state).tolist()

            if finish:
                if not replay:
                    Q_values[action] = reward
                    # Update network weights
                    model.update(state, Q_values)
                break

            if replay:
                # Update network weights using replay memory
                model.replay(memory, replay_size, gamma)
            else:
                # Update network weights using the last step only
                Q_values_next = model.predict(next_state)
                Q_values[action] = reward + gamma * torch.max(Q_values_next).item()
                model.update(state, Q_values)

            state = next_state

        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(reward_calc)
        if episode == (episodes - 1):
            ploter(final, title)
    return final


class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim=64, alpha=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim * 2, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), alpha)

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state))


class DQN_Replay_Memory(DQN):
    # Expand DQN class with a replay function
    def replay(self, memory, size, gamma=0.9):
        # Make sure the memory is big enough
        if len(memory) >= size:
            states = []
            targets = []
            # Sample a batch of experiences from the agent's memory
            batch = random.sample(memory, size)

            # Extract information from the data
            for state, action, next_state, reward, finish in batch:
                states.append(state)
                # Predict Q_values
                Q_values = self.predict(state).tolist()
                if finish:
                    Q_values[action] = reward
                else:
                    Q_values_next = self.predict(next_state)
                    Q_values[action] = reward + gamma * torch.max(Q_values_next).item()

                targets.append(Q_values)

            self.update(states, targets)


class DQN_double(DQN):
    def __init__(self, state_dim, action_dim, hidden_dim, alpha):
        super().__init__(state_dim, action_dim, hidden_dim, alpha)
        self.target = copy.deepcopy(self.model)

    def target_predict(self, s):
        with torch.no_grad():
            return self.target(torch.Tensor(s))

    def target_update(self):
        self.target.load_state_dict(self.model.state_dict())

    def replay(self, memory, size, gamma=1.0):
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract data points from the data
            for state, action, next_state, reward, finish in data:
                states.append(state)
                Q_values = self.predict(state).tolist()
                if finish:
                    Q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    Q_values_next = self.target_predict(next_state)
                    Q_values[action] = reward + gamma * torch.max(Q_values_next).item()

                targets.append(Q_values)

            self.update(states, targets)


def main():
    n_state = env.observation_space.shape[0]  # Number of states
    n_action = env.action_space.n  # Number of actions (left, right)
    episodes = 100  # Number of episodes
    n_hidden = 50  # Number of hidden nodes in the DQN
    alpha = 0.001  # Learning rate
    gamma = 0.9  # Gamma

    num_steps = 2  # number of simulation steps

    # for i in range(num_steps):
    #     clear_output(wait=True)
    #     env.reset()
    #     plt.figure()
    #     plt.imshow(get_screen().cpu().permute(1, 2, 0).numpy(),
    #                interpolation='none')
    #     plt.title('CartPole-v0 Environment')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.show()

    #####################################################################
    # # Get random search results
    # random_s = basic_search(env, episodes)

    # Get DQN results
    dqn = DQN(n_state, n_action, n_hidden, alpha)
    DeepQN = Q_Learning(env, dqn, episodes, gamma, epsilon=0.3)

    # # Get replay results
    # dqn_replay = DQN_Replay_Memory(n_state, n_action, n_hidden, alpha)
    # replayMemory = Q_Learning(env, dqn_replay, episodes, gamma, epsilon=0.2, replay=True,
    #                     title='DQN with Replay Memory')
    #
    # # Get double results
    # dqn_double = DQN_double(n_state, n_action, n_hidden, alpha)
    # doubleQ = Q_Learning(env, dqn_double, episodes, gamma, epsilon=0.2, replay=True, double=True,
    #                     title='Double DQN with Replay Memory', n_update=10)


if __name__ == "__main__":
    main()
