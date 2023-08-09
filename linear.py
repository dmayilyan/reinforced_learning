"""Linear QL agent"""
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

from config import *

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)

#################################
NUM_RUNS = 10
ALPHA = 0.1


def tuple2index(action_index: int, object_index: int) -> int:
    """Converts a tuple (a,b) to an index c"""
    return action_index * NUM_OBJECTS + object_index


def epsilon_greedy(
    state_vector: np.ndarray, theta: np.ndarray, epsilon: float
) -> Tuple[int, int]:
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector: extracted vector representation
        theta: current weight matrix
        epsilon: the probability of choosing a random command

    Returns:
        the indices describing the action/object to take
    """
    # Toss a coin to decide what to do
    if np.random.binomial(1, epsilon):
        # Randomly choose action and object
        action_index, object_index = np.random.randint(
            NUM_ACTIONS, size=1
        ), np.random.randint(NUM_OBJECTS, size=1)
    else:
        # Choose the best action and object
        action_index, object_index = np.unravel_index(
            np.argmax(theta @ state_vector), (NUM_ACTIONS, NUM_OBJECTS)
        )

    # if isinstance(action_index, np.ndarray):
    #     action_index = action_index[0]
    #
    # if isinstance(object_index, np.ndarray):
    #     object_index = object_index[0]

    return int(action_index), int(object_index)


def linear_q_learning(
    theta: np.ndarray,
    current_state_vector: np.ndarray,
    action_index: int,
    object_index: int,
    reward: float,
    next_state_vector: np.ndarray,
    terminal: bool,
) -> None:
    """Update theta for a given transition

    Args:
        theta: current weight matrix
        current_state_vector: vector representation of current state
        action_index: index of the current action
        object_index: index of the current object
        reward: the immediate reward the agent receives from playing current command
        next_state_vector: vector representation of next state
        terminal: True if this episode is over

    Returns:
        None
    """
    # If terminal step Q(s', c', theta) = 0
    if terminal:
        max_q = 0.0
    else:
        max_q = np.max(theta @ next_state_vector)

    # Q(s, c, theta) for current command, c
    q_val = (theta @ current_state_vector)[tuple2index(action_index, object_index)]

    # y = R(s, c) + gamma*maxQ
    y = reward + GAMMA * max_q

    theta[tuple2index(action_index, object_index)] = (
        theta[tuple2index(action_index, object_index)]
        + ALPHA * (y - q_val) * current_state_vector
    )


def run_episode(for_training: bool) -> None | float:
    """Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = 0.0  # initialize for each episode

    (current_room_desc, current_quest_desc, terminal) = framework.new_game()

    while not terminal:
        # Choose next action and execute
        current_state = current_room_desc + current_quest_desc
        current_state_vector = utils.extract_bow_feature_vector(
            current_state, dictionary
        )

        next_action_index, next_object_index = epsilon_greedy(
            current_state_vector, theta, epsilon
        )  # Get next action, object

        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            current_room_desc, current_quest_desc, next_action_index, next_object_index
        )  # Take a step

        next_state = next_room_desc + next_quest_desc  # Build next state vector
        next_state_vector = utils.extract_bow_feature_vector(next_state, dictionary)

        if for_training:
            # update Q-function.
            linear_q_learning(
                theta,
                current_state_vector,
                next_action_index,
                next_object_index,
                reward,
                next_state_vector,
                terminal,
            )  # Update theta

        if not for_training:
            # update reward
            epi_reward += (GAMMA ** (framework.STEP_COUNT - 1)) * reward

        # prepare next step
        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc

    if not for_training:
        return epi_reward


def run_epoch() -> np.ndarray:
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run() -> list:
    """Returns array of test reward per epoch for one run"""
    global theta
    theta = np.zeros([action_dim, state_dim])

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=200, ascii="░▒█")
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            f"Avg reward: {np.mean(single_run_epoch_rewards_test):.6f} | Ewma reward: {utils.ewma(single_run_epoch_rewards_test):.6f}"
        )

    return single_run_epoch_rewards_test


if __name__ == "__main__":
    data = utils.load_data("game.tsv")
    dictionary = utils.create_bag_of_words(data)
    state_dim = len(dictionary)
    action_dim = NUM_ACTIONS * NUM_OBJECTS

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(
        x, np.mean(epoch_rewards_test, axis=0)
    )  # plot reward per epoch averaged per run
    axis.set_xlabel("Epochs")
    axis.set_ylabel("reward")
    axis.set_title(
        f"Linear: nRuns={NUM_RUNS}, Epsilon={TRAINING_EP:.2f}, Epi={NUM_EPIS_TRAIN}, alpha={ALPHA:.4f}"
    )
    plt.show()
