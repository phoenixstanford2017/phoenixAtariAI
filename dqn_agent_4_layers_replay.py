# -*- coding: utf-8 -*-
import random
import logging
import numpy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

LOGGER = logging.getLogger(__name__)

EPISODES = 1000


class DQNAgent:
    def __init__(
            self,
            environment,
            action_space,
            eta=0.001,
            epsilon=1,
            discount=0.99,
            maxIters=10000
    ):
        """DQN agent builder
        :param environment:         the game environment object
        :param action_space:        the action space
        :param eta:                 the step of each gradient descent
        :param epsilon:             the probability of taking a random action
                                    instead of the optimal one
        :param discount:            discount factor for new state value
        :param maxIters:            Max number of iterations per episode
        """

        self.env = environment
        self.state_size = self.env.observation_space.shape[0]
        self.action_space = action_space
        self.action_space_n = len(action_space)
        self.eta = eta
        self.eps = epsilon
        self.discount = discount
        self.maxIters = maxIters
        self.eps_decaying_factor = 0.99999
        self.eta_decaying_factor = 0.999999
        self.min_eps = 0.05
        self.numIters = 0
        self.gameNumber=1
        self.inactivity_counter = 0
        self.tot_score = 0
        self.model = self.create_model()
        self.memory = deque(maxlen=100000)
        self.skip_counter = 0
        self.n_frame_skip = 4

    def create_model(self):
        """Method that creates
        the Neural Network model for DQN

        :return:    the compiled neural network
        """
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space_n, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.eta))
        return model

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

    def get_reward(self, reward, done, inactivity=False):
        if inactivity:
            if done:
                # Agent lost or lost life
                return -10
                # return -3000.0/self.tot_score

            if reward:
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
            else:
                self.tot_score += reward
        return reward

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, new_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.discount * numpy.amax(
                    self.model.predict(new_state))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def learn(self):
        state = self.env.reset()
        state = numpy.reshape(state, [1, self.state_size])
        for t in range(10000):
            self.numIters += 1

            # Frame skipping
            if not self.skip_counter:
                action = self.get_action(state)

            new_state, reward, done, debug_info = self.env.step(action)

            # Update tot score
            self.tot_score += reward

            # Reshape new state
            new_state = numpy.reshape(new_state, [1, self.state_size])


            # Frame skipping
            if not self.skip_counter:
                # Get reward
                reward = self.get_reward(reward, done, inactivity=True)


                # Stop if agent is stuck
                if self.inactivity_counter > 500:
                    # The agent is stuck
                    LOGGER.warning('The agent got stuck! reward: %s' % reward)
                    done = True

                self.remember(state, action, reward, new_state, done)
                if len(self.memory) > 30:
                    self.replay(30)

            if self.numIters % 10000 == 0:
                LOGGER.info(
                    'GameNumber:"%s" Iter "%s"',
                    self.gameNumber,
                    self.numIters,
                )
                LOGGER.info('epsilon: "%s", eta: "%s"', self.eps, self.eta)
                LOGGER.info('debugging info: %s', debug_info)
                LOGGER.info('saving the weights after %s iters' % self.numIters)
                self.save("./save_replay/phoenix-dqn-replay_%s.h5" % self.numIters)



            if self.skip_counter == self.n_frame_skip:
                self.skip_counter = 0
            else:
                self.skip_counter += 1

            if done:
                LOGGER.info('################# Game: %s finished score: %s', self.gameNumber, self.tot_score)
                # Reset counters
                self.gameNumber += 1
                self.inactivity_counter = 0
                self.tot_score = 0
                self.skip_counter = 0
                break

            # Initialize new state
            state = new_state
