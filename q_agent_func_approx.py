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
            epsilon=0.9,
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
        self.eps_decaying_factor = 0.99998
        self.eta_decaying_factor = 0.999999
        self.min_eps = 0.01
        self.numIters = 0
        self.weights = defaultdict(float)
        self.gameNumber=1
        self.inactivity_counter = 0
        self.tot_score = 0

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
        if self.eps and self.eps > self.min_eps:
            self.eps *= self.eps_decaying_factor
        # epsilon greedy.
        if numpy.random.random() <= self.eps:
            action = numpy.random.choice(self.action_space)
        else:
            action = max((self.getQ(state, action), action) for action in self.action_space)[1]
            # LOGGER.debug('Q_opt action: %s', action)
            # LOGGER.debug(list((self.getQ(state, action), action) for action in self.action_space))
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

        scalar = -self.eta*(Q_opt(state, action) - (reward+self.discount*V_opt(newState)))

        # Increment sparse vector
        for feature, f_value in self.feature_extractor(state, action).items():
            self.weights[feature] += scalar * f_value

    def get_reward(self, reward, done, inactivity=False):
        if inactivity:
            if done:
                # Agent lost or lost life
                return -10
                # return -3000.0/self.tot_score

            if reward:
                self.tot_score += reward
                reward *= 10
                self.inactivity_counter = 0
            elif not reward:
                # Agent did not hit anything
                if self.inactivity_counter % 10 == 0:
                    reward = -self.inactivity_counter
                # Increase the counter
                self.inactivity_counter += 1
        else:
            if done:
                reward = -10
            elif not reward:
                self.inactivity_counter += 1
                reward = -1
            else:
                self.tot_score += reward
        return reward

    def learn(self):
        state = self.env.reset()
        self.initial_state = state
        for t in range(10000):
            self.numIters += 1
            action = self.get_action(state)
            new_state, reward, done, debug_info = self.env.step(action)
            if self.numIters % 10000 == 0:
                LOGGER.info(
                    'GameNumber:"%s" Iter "%s"',
                    self.gameNumber,
                    self.numIters,
                )
                LOGGER.debug('epsilon: "%s", eta: "%s"', self.eps, self.eta)
                LOGGER.debug('weights vector lenght: %s', len(self.weights))
                LOGGER.debug('debugging info: %s', debug_info)

            # self.eta *= self.eta_decaying_factor

            # Get reward
            reward = self.get_reward(reward, done, inactivity=True)


            # Stop if agent is stuck
            if self.inactivity_counter > 500:
                # The agent is stuck
                LOGGER.warning('The agent got stuck!')
                done = True


            self.incorporateFeedback(state, action, reward, new_state, done)
            if done:
                LOGGER.debug('################# Game: %s finished score: %s', self.gameNumber, self.tot_score)
                # Reset counters
                self.gameNumber += 1
                self.inactivity_counter = 0
                self.tot_score = 0
                break

            # Initialize new state
            state = new_state