import gym
import logging
logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)

env = gym.make('Phoenix-ram-v0')

tot_rewards = []
quiet = False
for i_episode in range(2):
    observation = env.reset()
    tot_reward = 0
    print '##################'
    for t in range(10000):
        if not quiet:
            env.render()
        action = env.action_space.sample()
        #print 'action: %s' % action
        observation2, reward, done, info = env.step(action)
        if reward:
            #print observation
            print info
            print reward
            tot_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Final Score: %s" % tot_reward)
            tot_rewards.append(tot_reward)
            break

print "random agent average: %s" % (sum(tot_rewards)/float(len(tot_rewards)))
