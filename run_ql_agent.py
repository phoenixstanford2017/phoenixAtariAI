import argparse
import os
import gym
import logging
from ql_agent import QAgentFuncApprox

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

LOGGER.setLevel(logging.INFO)


def parse_args():
    """
    Simple parser function that returns the parsed args
    """
    pars = argparse.ArgumentParser()
    pars.add_argument(
        '--learning',
        '-l',
        dest='learning',
        action='store_true',
        default=False,
        help=(
            'Switch to learning mode. The agent will learn from the game '
            'instead of simply playing.'
        )
    )
    pars.add_argument(
        '--maxiters',
        '-m',
        dest='max_iters',
        type=int,
        default=10**5,
        help=(
            'The number of maximum iterations.'
        )
    )
    pars.add_argument(
        '--weights_dir',
        '-wdir',
        type=str,
        dest='weights_dir',
        default='weights_dir',
        help=(
            'The weights directory.'
        )
    )
    pars.add_argument(
        '--load',
        dest='load',
        type=str,
        required=False,
        help=(
            'The path of the weights file to load. When run in playing mode.'
        )
    )
    pars.add_argument(
        '--no-quiet',
        '-nq',
        dest='quiet',
        action='store_false',
        default=True,
        help=(
            'Run in not quiet mode, enable game graphic.'
        )
    )
    pars.add_argument(
        '--debug',
        '-D',
        dest='debug',
        action='store_true',
        default=False,
        help=(
            'Run this script in debug mode'
        )
    )
    return pars.parse_args()


def training(**kwargs):
    # Set logging level
    if kwargs['debug']:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)


    agent = QAgentFuncApprox(
        environment=env,
        action_space=[0, 1, 2, 3, 4, 5, 6, 7],
        epsilon=0.4,
        eta=0.001,
        discount=0.95,
        maxIters=kwargs['max_iters']
    )
    while True:
        agent.learn()
        if agent.numIters > agent.maxIters:
            break

    agent.writingWeights('%s/weights_it%s.txt' % (kwargs['weights_dir'], kwargs['max_iters']))

    return agent


def set_up_weights_dir(dir_path):
    """ Function that set up the dqn agent weight directory

    :param dir_path:    path to the weights directory
    :return:            None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_scores_to_csv(csv_file_path, game_num, score, num_timesteps):
    with open(csv_file_path, 'a') as csv_file:
        csv_file.write("%s, %s, %s\n" % (game_num, score, num_timesteps))


def play(environment, agent, quiet=False):
    # Set agent epsilon to 0
    agent.eps = 0

    LOGGER.setLevel(logging.INFO)
    tot_rewards = []
    for i_episode in range(100):
        obs = environment.reset()
        tot_reward = 0
        for t in range(5000):
            if not quiet:
                environment.render()
            action = agent.get_action(obs)
            obs2, reward, done, info = environment.step(action)
            if reward:
                tot_reward += reward
            if done:
                print("Episode {1} finished after {0} timesteps".format((t + 1), i_episode+1))
                print("Final Score: %s" % tot_reward)
                tot_rewards.append(tot_reward)
                break
            # Reset the state
            obs = obs2
        else:
            print("Episode {1} finished after {0} timesteps".format((5000), i_episode+1))
            print("Final Score: %s" % tot_reward)
            tot_rewards.append(tot_reward)

    print "agent score average: %s" % (
        sum(tot_rewards) / float(len(tot_rewards))
    )


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # set up weights dir
    set_up_weights_dir(args.weights_dir)

    # Train and play or just play
    if args.learning:
        LOGGER.info(
            'Running Q-Learning agent with function approx in learning mode.'
        )
        print "#############\nlearning...with following args: %s\n#############" % vars(args)
        trained_agent = training(**vars(args))
    else:
        LOGGER.info(
            'Running Q-Learning agent with function approx in playing mode.'
        )
        trained_agent = QAgentFuncApprox(environment=env, action_space=[0, 1, 2, 3, 4, 5, 6, 7], epsilon=0)
        if args.load:
            trained_agent.readingWeights(args.load)
        else:
            LOGGER.error(
                'No weights file specified! Please specify a weight file when '
                'running in playing mode. '
                'E.g: --load dqn_weights/weights_file.h5'
            )
            exit(1)

    play(environment=env, agent=trained_agent, quiet=args.quiet)
