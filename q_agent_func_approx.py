from collections import defaultdict
import logging
import numpy
from os.path import expanduser
import pickle

LOGGER = logging.getLogger(__name__)


class QAgentFuncApprox(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(
            self,
            environment,
            action_space,
            eta=0.02,
            epsilon=0.99,
            discount=0.9,
            maxIters=10000
    ):
        """Q-learning agent with function approx builder
        :param environment:         the game environment object
        :param action_space:        the action space
        :param eta:                 the step of each gradient descent
        :param epsilon:             the probability of taking a random action
                                    instead of the optimal one
        :param discount:            discount factor for new state value
        :param maxIters:            Max number of iterations per episode
        """
        self.env = environment
        self.action_space = action_space
        self.action_n = len(action_space)
        self.eta = eta
        self.eps = epsilon
        self.discount = discount
        self.maxIters = maxIters
        self.eps_decaying_factor = 0.99999
        self.eta_decaying_factor = 0.99999
        self.numIters = 0
        self.weights = defaultdict(float)
        self.gameNumber=1

    def readingWeights(self, filePath):
        with open(filePath, 'rb') as handle:
            self.weights = pickle.loads(handle.read())

    def writingWeights(self, filePath):
        with open(filePath, 'wb') as handle:
            pickle.dump(self.weights, handle)

    def feature_extractor(self, state, action):
        """ Method returning the feature vector phi(s,a)
            given a specific game state and an action

        :param state:       current state of the game
        :param action:      action taken in the current state
        :return:            a default dictionary containing
                            feature:feature_value pairs
        """
        phi = defaultdict(float)
        # phi[(tuple(state),action)] = 1
        for i, v in enumerate(state):
            phi[(i, v, action)] = 1
        # LOGGER.debug('phi for iteration: %s new phi is: %s', self.numIters, phi)
        return phi

        # phi=[]
        # featureValue = 1
        # fk1 = (tuple(state), action)
        #
        # phi.append((fk1,featureValue))
        # for i, v in enumerate(state):
        #     fk2=(i,v,action)
        #     phi.append((fk2,featureValue))
        # #print('phi for iteration: %s new phi is: %s' % (self.numIters, phi))
        # return phi

    def getQ(self, state, action):
        """ Method returning the dot product of the feature vector phi(s,a)
            and the weight vector

        :param state:       current state of the game
        :param action:      action taken in the current state
        :return:            The Q_opt of state and action
        """
        Q_opt = 0
        for feature, value in self.feature_extractor(state, action).items():
            Q_opt += self.weights[feature] * value
        return Q_opt

    def get_eta(self):
        """ Method which returns a Stepsize inversely proportional
            to the number of total iterations during the overall training

        :return:    the original eta divided by the sqare root of the number of
                    iterations
        """
        return self.eta / numpy.sqrt(self.numIters)

    def get_action(self, state):
        """Method that perform epsilon-greedy action selection: given a state
            returns either a random action or the arg max of the Q_opt-value

        :param state:       current state of the game
        :return:            action from the action space
        """
        if self.eps:
            self.eps *= self.eps_decaying_factor
        # epsilon greedy.
        if numpy.random.random() <= self.eps:
            action = numpy.random.choice(self.action_space)
        else:
            action = max((self.getQ(state, action), action) for action in self.action_space)[1]
            LOGGER.debug('Q_opt action: %s', action)
            LOGGER.debug(list((self.getQ(state, action), action) for action in self.action_space))
        return action

    def incorporateFeedback(self, state, action, reward, newState, done=False):
        """Method that updates the weight vector based on state
            action and reward

        :param state:           current state of the game
        :param action:          action taken in the current state
        :param reward:          reward obtained transitioning in the new state
        :param newState:        new state of the game
        :param done:            boolean representing isEnd() function
        :return:                None
        """
        Q_opt = lambda s, a: self.getQ(s, a)
        V_opt = lambda s: 0 if s is done else max(self.getQ(s, a) for a in self.action_space)

        # self.eta = self.get_eta()
        self.eta *= self.eta_decaying_factor
        reward = -20 if done else reward
        reward *= 10
        if not reward:
            reward = -3

        scalar = -self.eta*(Q_opt(state, action) - (reward+self.discount*V_opt(newState)))

        # Increment sparse vector
        for feature, f_value in self.feature_extractor(state, action).items():
            self.weights[feature] += scalar * f_value

    def learn(self):
        state = self.env.reset()
        for t in range(self.maxIters):
            self.numIters += 1
            action = self.get_action(state)
            new_state, reward, done, debug_info = self.env.step(action)
            if self.numIters % 500 == 0:
                LOGGER.info(
                    'GameNumber:"%s" Iter "%s"',
                    self.gameNumber,
                    self.numIters,
                )
                LOGGER.info('epsilon: "%s", eta: "%s"', self.eps, self.eta)
                # LOGGER.debug('weights vector: %s', self.weights)
                LOGGER.info('debugging info: %s', debug_info)

            self.incorporateFeedback(state, action, reward, new_state, done)
            if done:
                self.gameNumber += 1
                break

            # Initialize new state
            state = new_state