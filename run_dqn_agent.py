import argparse
import logging
import os

import gym
import numpy

from dqn import DQNAgent

# Set up logging
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)
# Set level and format for cli and core console logging.
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
            'Switch to learning model. The agent will learn from the game '
            'instead of simply playing.'
        )
    )
    pars.add_argument(
        '--architecture',
        '-a',
        dest='NN_arch',
        type=str,
        required=True,
        choices=['3l', '4l'],
        help=(
            'The NN architecture for the DQN agent: 3 layers or 4layers.'
        )
    )
    pars.add_argument(
        '--maxiters',
        '-m',
        dest='max_iters',
        type=int,
        default=10**6,
        help=(
            'The number of maximum iterations.'
        )
    )
    pars.add_argument(
        '--replay',
        '-r',
        action='store_true',
        default=False,
        help=(
            'Switch to use replay memory technique.'
        )
    )
    pars.add_argument(
        '--batch_size',
        '-bs',
        type=int,
        default=30,
        help=(
            'The size of the minibatch sampled from the agent memory '
            'if using replay memory technique.'
        )
    )
    pars.add_argument(
        '--frame_skipping',
        '-fs',
        type=int,
        default=0,
        help=(
            'Use frame skipping. Specify how many steps you want to skip.'
        )
    )
    pars.add_argument(
        '--weights_dir',
        '-wdir',
        type=str,
        default='dqn_weights',
        help=(
            'Use frame skipping. Specify how many steps you want to skip.'
        )
    )
    pars.add_argument(
        '--load',
        dest='load',
        type=str,
        required=False,
        help=(
            'The path of the file to load.'
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

    agent = DQNAgent(
        environment=env,
        action_space=[0, 1, 2, 3, 4, 5, 6, 7],
        NN_arch=kwargs['NN_arch'],
        maxIters=kwargs['max_iters'],
        eta=0.00001,
        epsilon=0.4,
        discount=0.95,
        weights_dir=kwargs['weights_dir']
    )

    while True:
        agent.learn(
            replay=kwargs['replay'],
            frame_skipping=kwargs['frame_skipping'],
            batch_size=kwargs['batch_size']
        )
        if agent.numIters > agent.maxIters:
            break

    agent.save(agent.save_path % kwargs['max_iters'])
    # return the agent object
    return agent


def save_scores_to_csv(csv_file_path, game_num, score, num_timesteps):
    with open(csv_file_path, 'a') as csv_file:
        csv_file.write("%s, %s, %s\n" % (game_num, score, num_timesteps))


def set_up_weights_dir(dir_path):
    """ Function that set up the dqn agent weight directory

    :param dir_path:    path to the weights directory
    :return:            None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def play(environment, agent, quiet=False):
    # Set agent epsilon to 0
    agent.eps = 0
    LOGGER.setLevel(logging.INFO)

    tot_rewards = []
    for i_episode in range(100):
        obs = env.reset()
        obs = numpy.reshape(obs, [1, agent.state_size])
        tot_reward = 0
        for t in range(5000):
            if not quiet:
                env.render()
            action = agent.get_action(obs)
            obs2, reward, done, info = env.step(action)
            obs2 = numpy.reshape(obs2, [1, agent.state_size])
            if reward:
                tot_reward += reward
            if done:
                print("Episode {1} finished after {0} timesteps".format((t + 1), i_episode))
                print("Final Score: %s" % tot_reward)
                tot_rewards.append(tot_reward)
                break
            # Reset the state
            obs = obs2
        else:
            print("Episode {1} finished after {0} timesteps".format((5000), i_episode))
            print("Final Score: %s" % tot_reward)
            tot_rewards.append(tot_reward)

        # csv_file_path = 'scores/%s_dqn_scores.csv' % 900000
        # save_scores_to_csv(csv_file_path, i_episode, tot_reward, t)

    print "agent score average: %s" % (
    sum(tot_rewards) / float(len(tot_rewards)))




if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # set up weights dir
    set_up_weights_dir(args.weights_dir)

    # Train and play or just play
    if args.learning:
        print "#############\nlearning...with following args: %s\n#############" % vars(args)
        trained_agent = training(**vars(args))
    else:
        trained_agent=DQNAgent(environment=env, action_space=[0, 1, 2, 3, 4, 5, 6, 7], epsilon=0, NN_arch=args.NN_arch)
        if args.load:
            trained_agent.load(args.load)
        else:
            LOGGER.warning(
                'No weights specified the agent will play without being trained'
            )
    play(environment=env, agent=trained_agent, quiet=True)



