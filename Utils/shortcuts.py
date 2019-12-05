import math
import sys
import logging
from tqdm import tqdm
from tqdm import trange
import numpy as np


def get_logger(logger_name=None):
    if logger_name is not None:
        logger = logging.getLogger(logger_name)
        logger.propagate = 0
    else:
        logger = logging.getLogger("taufikxu")
    return logger


def channel_last(array_or_list, is_array=True):
    if is_array is False:
        img = array_or_list
        if img.shape[-1] <= 4:
            return img
        else:
            return np.transpose(img, [1, 2, 0])

    img = array_or_list[0]
    if img.shape[-1] <= 4:
        return array_or_list

    if isinstance(array_or_list, list) is not True:
        array_or_list = np.transpose(array_or_list, [0, 2, 3, 1])
    else:
        array_or_list = [
            np.transpose(item, [1, 2, 0]) for item in array_or_list
        ]
    return array_or_list


class random_seed:
    seed = 1

    def __init__(self):
        pass

    @classmethod
    def get_seed(cls):
        cls.seed += 1
        return cls.seed

    @classmethod
    def set_seed(cls, _seed):
        cls.seed = _seed


def seed():
    return random_seed.get_seed()


def xrange(iters, prefix=None, Epoch=None, **kwargs):
    if Epoch is not None and prefix is None:
        prefix = "Epoch " + str(Epoch)
    return trange(int(iters),
                  file=sys.stdout,
                  leave=False,
                  dynamic_ncols=True,
                  desc=prefix,
                  **kwargs)


class processbar(object):
    def __init__(self, iters, epoch=0, **kwargs):
        self.tqdm = None
        self.iters = iters
        self.epoch = epoch
        self.current = 0
        self.kwargs = kwargs

    def check(self):
        if self.tqdm is None:
            self.reset()

    def reset(self):
        if self.tqdm is not None:
            self.tqdm.close()

        prefix = "Epoch " + str(self.epoch)
        self.tqdm = tqdm(
            range(int(self.iters)),
            file=sys.stdout,
            leave=False,
            dynamic_ncols=True,
            desc=prefix,
        )
        self.current = 0

    def update(self, n):
        self.check()
        if self.current >= self.iters:
            self.reset()

        self.current += n
        self.tqdm.update(n)

    def next(self):
        self.check()
        self.update(1)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
