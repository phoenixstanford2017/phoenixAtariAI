from collections import defaultdict
import gym.spaces
env = gym.make('Phoenix-ram-v0')
print env.observation_space
print env.action_space
    
import   q_agent_func_approx 
learner= q_agent_func_approx.QAgentFuncApprox(env.observation_space,[0,1,2,3,4])
for ind in range(10):
    learner.learn(env)
    print "new game:%d" %ind
qOpt = defaultdict()

'''
for observation in env.observation_space:
    action=learner.get_action(observation,eps=0)
    qOpt[observation]=action
'''


tot_rewards = []
quiet = False
for i_episode in range(2):
    observation = env.reset()
    tot_reward = 0
    print '##################'
    for t in range(10000):
        if not quiet:
            env.render()
        #action = qOpt[observation]
        action=learner.get_action(observation, eps=0)
        #print 'action: %s' % action
        observation, reward, done, info = env.step(action)
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
