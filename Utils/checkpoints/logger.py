import os
import time
import logging
import operator
import coloredlogs

from Utils.shortcuts import get_logger

from matplotlib import pyplot as plt

plt.switch_backend("Agg")


def build_logger(folder=None, args=None, logger_name=None):
    FORMAT = "%(asctime)s;%(levelname)s|%(message)s"
    DATEF = "%H-%M-%S"
    logging.basicConfig(format=FORMAT)
    logger = get_logger(logger_name)
    # logger.setLevel(logging.DEBUG)

    if folder is not None:
        fh = logging.FileHandler(filename=os.path.join(
            folder, "logfile{}.log".format(time.strftime("%m-%d"))))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s;%(levelname)s|%(message)s",
                                      "%H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    LEVEL_STYLES = dict(
        debug=dict(color="magenta"),
        info=dict(color="green"),
        verbose=dict(),
        warning=dict(color="blue"),
        error=dict(color="yellow"),
        critical=dict(color="red", bold=True),
    )
    coloredlogs.install(level=logging.INFO,
                        fmt=FORMAT,
                        datefmt=DATEF,
                        level_styles=LEVEL_STYLES)

    def get_list_name(obj):
        if type(obj) is list:
            for i in range(len(obj)):
                if callable(obj[i]):
                    obj[i] = obj[i].__name__
        elif callable(obj):
            obj = obj.__name__
        return obj

    if isinstance(args, dict) is not True:
        args = vars(args)

    sorted_list = sorted(args.items(), key=operator.itemgetter(0))
    logger.info("#" * 120)
    logger.info("----------Configurable Parameters In this Model----------")
    for name, val in sorted_list:
        logger.info("# " + ("%20s" % name) + ":\t" + str(get_list_name(val)))
    logger.info("#" * 120)
    return logger
