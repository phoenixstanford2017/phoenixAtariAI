import cPickle as pickle
import gym
import logging
import datetime
from q_agent_func_approx import QAgentFuncApprox
import sys, getopt

# Set up logging
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)
# Set level and format for cli and core console logging.
# create logger with 'spam_application'
LOGGER.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
LOGGER.addHandler(ch)

# Invoke environment
env = gym.make('Phoenix-ram-v0')

LOGGER.setLevel(logging.DEBUG)
RECORD_SCORE = 1733

def training(weightFile, maxIters):
    agent = QAgentFuncApprox(environment=env, action_space=[0, 1, 2, 3, 4, 5, 6, 7], epsilon=0.4, eta=0.001, discount=0.95, maxIters=maxIters)
    while True:
        # agent.readingWeights('weights_files/weights_1511.txt')
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
    for i_episode in range(100):
        obs = env.reset()
        tot_reward = 0
        for t in range(5000):
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
        else:
            print("Episode finished after {} timesteps".format(5000))
            print("Final Score: %s" % tot_reward)
            tot_rewards.append(tot_reward)

            # if tot_reward > RECORD_SCORE:
            #     agent.writingWeights('weights_files/weights_%s.txt' % int(tot_reward))

    print "agent score average: %s" % (
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
    for i in range(9, 10):
        if learning:
            print "#############\nlearning...with maxiter: %s\n#############" % (i*10000)
            trained_agent = training(weightFile, i*10000)
        else:
            trained_agent=QAgentFuncApprox(environment=env, action_space=[0, 1, 2, 3, 4, 5, 6, 7], epsilon=0)
            weightFile = "weights_files/weights_2960.txt"
            trained_agent.readingWeights(weightFile)


        play(environment=env, agent=trained_agent, quiet=False)



