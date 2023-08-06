from prettyconf import config

from prettyconf.loaders import IniFile
import os

ini_path = os.path.join(os.path.dirname(__file__), "config.ini")

config.loaders = [IniFile(filename=ini_path)]

DEBUG = config("debug", default=False, cast=config.boolean)

GAMMA = config("gamma", cast=config.eval)  # discounted factor
TRAINING_EP = config(
    "TRAINING_EP", cast=config.eval
)  # epsilon-greedy parameter for training
TESTING_EP = config(
    "TESTING_EP", cast=config.eval
)  # epsilon-greedy parameter for testing
NUM_RUNS = config("NUM_RUNS", default=5)
NUM_EPOCHS = config("NUM_EPOCHS", cast=config.eval)
NUM_EPIS_TRAIN = config(
    "NUM_EPIS_TRAIN", cast=config.eval
)  # number of episodes for training at each epoch
NUM_EPIS_TEST = config(
    "NUM_EPIS_TEST", cast=config.eval
)  # number of episodes for testing
ALPHA = config("ALPHA", default=0.01, cast=config.eval)  # learning rate for training
