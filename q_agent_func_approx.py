from collections import defaultdict
import numpy
from gym.spaces import discrete

class UnsupportedSpace(Exception):
    pass

class QAgentFuncApprox(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, discrete.Discrete):
            raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        if not isinstance(action_space, discrete.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "eta" : 0.1,
            "eps": 0.05,            # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_iter": 10000}        # Number of iterations
        self.config.update(userconfig)
        # TODO define the feature extractor

        # TODO define Q_opt as w dotproduct phi(s,a) (similar to get_Qopt in blackjack 4a assignment)
        self.q = defaultdict(lambda: self.config["init_std"] * numpy.random.randn(self.action_n) + self.config["init_mean"])

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        # TODO update the argmax argument based on the new Q_opt function approx definition
        action = numpy.argmax(self.q[observation.item()]) if numpy.random.random() > eps else self.action_space.sample()
        return action

    def learn(self, env):
        config = self.config
        obs = env.reset()
        q = self.q
        for t in range(config["n_iter"]):
            action, _ = self.act(obs)
            obs2, reward, done, _ = env.step(action)
            future = 0.0
            # TODO Update the incorporate feedback based on Q-learning with function approx algorithm
            if not done:
                future = numpy.max(q[obs2.item()])
            q[obs.item()][action] -= \
                self.config["eta"] * (q[obs.item()][action] - reward - config["discount"] * future)

            obs = obs2