import cPickle as pickle
import gym
import logging
import datetime
from q_agent_func_approx import QAgentFuncApprox

# Set up logging
formatter = '%(asctime)s %(levelname)s %(name)s %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)
LOGGER = logging.getLogger(__name__)

# Invoke environment
env = gym.make('Phoenix-ram-v0')


def training():
    agent = QAgentFuncApprox(environment=env, action_space=[0, 1, 2, 3, 4], epsilon=0.4)
    for i_episode in range(100):
        agent.learn()

    # Save the weight vector
    # timestamp = datetime.datetime.now().isoformat()
    # with open('weights_files/weights_%s.p' % timestamp, 'wb') as fp:
    #     pickle.dump(agent.weights, fp)

    # return the agent object
    return agent


def play(environment, agent, quiet=False):
    # Set agent epsilon to 0
    agent.eps = 0

    LOGGER.setLevel(logging.DEBUG)
    tot_rewards = []
    for i_episode in range(10):
        obs = env.reset()
        tot_reward = 0
        for t in range(10000):
            if not quiet:
                env.render()
            action = agent.get_action(obs)
            obs2, reward, done, info = env.step(action)
            if reward:
                tot_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                print("Final Score: %s" % tot_reward)
                tot_rewards.append(tot_reward)
                break
            # Reset the state
            obs = obs2

    print "random agent average: %s" % (
    sum(tot_rewards) / float(len(tot_rewards)))

if __name__ == '__main__':
    trained_agent = training()
    play(environment=env, agent=trained_agent)



