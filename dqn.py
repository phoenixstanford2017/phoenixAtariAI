# -*- coding: utf-8 -*-
import random
import logging
import numpy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

LOGGER = logging.getLogger(__name__)


class InvalidNNArch(Exception):
    pass


class DQNAgent:
    def __init__(
            self,
            environment,
            action_space,
            NN_arch,
            weights_dir='dqn_weights',
            eta=0.001,
            epsilon=1,
            discount=0.99,
            maxIters=10000,
            mem_size=100000,
    ):
        """DQN agent builder
        :param environment:         the game environment object
        :param action_space:        the action space
        :param eta:                 the step of each gradient descent
        :param NN_arch:             NN architecture, either 3l or 4l (l = layers)
        :param weights_dir:         The directory containing the agent weights
        :param epsilon:             the probability of taking a random action
                                      instead of the optimal one
        :param discount:            discount factor for new state value
        :param maxIters:            Max number of iterations per episode
        :param mem_size:            The size of the replay memory
        """

        self.env = environment
        self.state_size = self.env.observation_space.shape[0]
        self.action_space = action_space
        self.action_space_n = len(action_space)
        self.memory = deque(maxlen=mem_size)

        # Hyperparameters
        self.eta = eta
        self.eps = epsilon
        self.discount = discount
        self.maxIters = maxIters
        self.eps_decaying_factor = 0.99999
        self.min_eps = 0.05

        # Counters
        self.numIters = 0
        self.gameNumber=1
        self.inactivity_counter = 0

        # Misc
        self.save_path = self._create_save_path_name(weights_dir, NN_arch)

        # Neural Network
        if NN_arch == '3l':
            self.model = self.create_3_layer_model()
        elif NN_arch == '4l':
            self.model = self.create_4_layer_model()
        else:
            raise InvalidNNArch

    def _create_save_path_name(self, weights_dir, NN_arch):
        weights_file_name = (
            'dqn_%s_eta_%s_disc_%s_eps_%s_mineps_%s_deceps_%s_it_%%s.h5' %
            (NN_arch,
             self.eta,
             self.discount,
             self.eps,
             self.min_eps,
             self.eps_decaying_factor)
        )
        LOGGER.info(
            'The agent weights will be saved in "%s", with format "%s"'
            % (weights_dir, weights_file_name)
        )
        return '/'.join([weights_dir, weights_file_name])

    def create_3_layer_model(self):
        """Method that creates
        the 3 layer Neural Network model for DQN

        :return:    the compiled neural network
        """
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_space_n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.eta))
        return model

    def create_4_layer_model(self):
        """Method that creates
        the 4 layer Neural Network model for DQN

        :return:    the compiled neural network
        """
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space_n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.eta))
        return model

    def load(self, name):
        """Method that loads NN weights

        :param name:    weights file path
        :return:        None
        """
        self.model.load_weights(name)

    def save(self, name):
        """Method that saves NN weights

        :param name:    weights file path
        :return:        None
        """
        self.model.save_weights(name)

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
            LOGGER.debug('random action')
        else:
            act_values = self.model.predict(state)
            LOGGER.debug('action values %s' % act_values)
            LOGGER.debug('action values %s' % numpy.argmax(act_values[0]))
            return numpy.argmax(act_values[0])

        return action

    def get_reward(self, reward, done):
        """Methods that modify rewards based on game status

        :param reward:      reward returned by the environment
        :param done:        boolean, ifEnd(s)
        :return:            modified reward
        """
        if done:
            # Agent lost or lost life
            return -10
        if reward:
            reward *= 10
            self.inactivity_counter = 0
        elif not reward:
            # Agent did not hit anything
            if self.inactivity_counter % 10 == 0:
                reward = -self.inactivity_counter
            # Increase the counter
            self.inactivity_counter += 1

        return reward

    def fit_neural_network(self, state, action, reward, new_state, done):
        """Method that tunes the weights using the obtained s,a,r,s'
            from the environment

        :param state:       game state
        :param action:      agent action
        :param reward:      reward returned with the new state
        :param new_state:   new game state
        :param done:        boolean, ifEnd(s)
        :return:            None
        """
        target = reward
        if not done:
            target = reward + self.discount * numpy.amax(
                self.model.predict(new_state))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, new_state, done):
        """Method caching the s,a,r,s' for performing memory replay

        :param state:       game state
        :param action:      agent action
        :param reward:      reward returned with the new state
        :param new_state:   new game state
        :param done:        boolean, ifEnd(s)
        :return:            None
        """
        self.memory.append((state, action, reward, new_state, done))

    def replay(self, batch_size):
        """Method that tunes the weights of the NN replaying randomly selected
            s,a,r,s' from the cache

        :param batch_size:      Size of the random sample extracted
                                  from agent memory
        :return:                None
        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, new_state, done in minibatch:
            self.fit_neural_network(state, action, reward, new_state, done)

    def learn(self, replay=False, frame_skipping=0, batch_size=30):
        """ Method that runs the training episodes

        :param replay:              boolean for activating memory replay
        :param frame_skipping:      number of frames to skip during training
        :return:    None
        """
        tot_score = 0
        state = self.env.reset()
        state = numpy.reshape(state, [1, self.state_size])
        for t in range(10000):
            self.numIters += 1

            # Get action
            action = self.get_action(state)
            # Get new state
            new_state, reward, done, debug_info = self.env.step(action)

            # Update tot score
            tot_score += reward

            # Reshape new state
            new_state = numpy.reshape(new_state, [1, self.state_size])

            # Frame skipping
            # Get reward
            reward = self.get_reward(reward, done)

            # Stop if agent is stuck
            if self.inactivity_counter > 500:
                # The agent is stuck
                LOGGER.warning('The agent got stuck! reward: %s' % reward)
                done = True

            # Incorporate feedback
            if replay:
                self.remember(state, action, reward, new_state, done)
                if len(self.memory) > batch_size:
                    self.replay(batch_size)
            else:
                self.fit_neural_network(state, action, reward, new_state, done)

            # Frame skipping
            if not done:
                for i in range(frame_skipping):
                    new_state, reward, done, debug_info = self.env.step(action)
                    # Update tot score
                    tot_score += reward
                    # Reshape new state
                    new_state = numpy.reshape(new_state, [1, self.state_size])
                    if done:
                        break

            if self.numIters % 10000 == 0:
                LOGGER.info(
                    'GameNumber:"%s" Iter "%s"',
                    self.gameNumber,
                    self.numIters,
                )
                LOGGER.info('epsilon: "%s", eta: "%s"', self.eps, self.eta)
                LOGGER.info('debugging info: %s', debug_info)
                LOGGER.info('saving the weights after %s iters' % self.numIters)
                self.save(self.save_path % self.numIters)

            if done:
                LOGGER.info('################# Game: %s finished score: %s', self.gameNumber, tot_score)
                # Reset counters
                self.gameNumber += 1
                self.inactivity_counter = 0
                break

            # Initialize new state
            state = new_state
