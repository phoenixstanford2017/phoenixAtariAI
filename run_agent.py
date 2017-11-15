import cPickle as pickle
import gym
import logging
import datetime
from q_agent_func_approx import QAgentFuncApprox
import sys, getopt

# Set up logging
formatter = '%(asctime)s %(levelname)s %(name)s %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)
LOGGER = logging.getLogger(__name__)

# Invoke environment
env = gym.make('Phoenix-ram-v0')


def training(weightFile, nEpisodes=100):
    agent = QAgentFuncApprox(environment=env, action_space=[0, 1, 2, 3, 4, 5, 6, 7], epsilon=0.4, eta=0.02, maxIters=50000)
    for i_episode in range(nEpisodes):
        agent.learn()
        if agent.numIters > agent.maxIters:
            break

    # Save the weight vector
    agent.writingWeights(weightFile)
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
   
    weightFileCommanline = None
    learning=False
    nEpisodes=100
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hlw:n:",["wfile="])
    except getopt.GetoptError:
        print 'run_agent.py -w <weightFile> -l'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'run_agent.py -w <weightFile> -l'
            sys.exit()
        elif opt in ("-w","-wfile"):
            weightFileCommanline = arg
        elif opt in ("-l"):
            learning=True
        elif opt in ("-n"):
            nEpisodes=int(arg)
    if weightFileCommanline != None:
        weightFile = weightFileCommanline
    else:
        #default
        weightFile="weights_files/weights.txt"

    print 'weight file is ', weightFile
    if  learning:
        print "learning..." 
        trained_agent = training(weightFile,nEpisodes)
    else:
        trained_agent=QAgentFuncApprox(environment=env, action_space=[0, 1, 2, 3, 4], epsilon=0.4)
        trained_agent.readingWeights(weightFile)


    play(environment=env, agent=trained_agent)



