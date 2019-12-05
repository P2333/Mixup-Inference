import os
import numpy as np
from matplotlib import pyplot as plt

plt.switch_backend("Agg")


class MetricSaver(object):
    def __init__(self,
                 name,
                 path,
                 ma_weight=0.9,
                 figsize=(6, 4),
                 postfix=".png",
                 save_on_update=True):
        self.name = name
        self.path = path
        self.ma_weight = ma_weight

        self.step = [0.]
        self.original_value = []
        self.value = [0.]
        self.ma_value = [0.]

        self.figsize = figsize
        self.postfix = postfix
        self.save_on_update = save_on_update

    def _test_valid_step(self):
        for i in range(1, len(self.step)):
            if self.step[i] is None:
                return False
            if self.step[i] < self.step[i - 1]:
                return False
        return True

    def update(self, step=None, value=None, save=True):
        self.step.append(step)
        self.original_value.append(value)

        value = np.mean(value)
        self.value.append(value)
        if len(self.ma_value) == 1:
            self.ma_value.append(value)
        else:
            self.ma_value.append(self.ma_value[-1] * self.ma_weight + value *
                                 (1. - self.ma_weight))

        if save is True or self.save_on_update is True:
            self.save()

    def save(self):
        if self._test_valid_step() is True:
            fig = plt.figure(figsize=self.figsize)
            plt.plot(
                self.step[1:],
                self.value[1:],
                '-',
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, self.name + self.postfix),
                        bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=self.figsize)
            plt.plot(
                self.step[1:],
                self.ma_value[1:],
                '-',
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.path,
                                     self.name + "_ma" + self.postfix),
                        bbox_inches='tight')
            plt.close(fig)
        else:
            fig = plt.figure(figsize=self.figsize)
            plt.plot(
                self.value[1:],
                '-',
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.path, self.name + self.postfix),
                        bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=self.figsize)
            plt.plot(
                self.ma_value[1:],
                '-',
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.path,
                                     self.name + "_ma" + self.postfix),
                        bbox_inches='tight')
            plt.close(fig)
        np.savez(os.path.join(self.path, self.name + ".npz"),
                 value=self.value[1:],
                 ma_value=self.ma_value,
                 original_value=self.original_value)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {
            name + postfix: meter.val
            for name, meter in self.meters.items()
        }

    def averages(self, postfix='/avg'):
        return {
            name + postfix: meter.avg
            for name, meter in self.meters.items()
        }

    def sums(self, postfix='/sum'):
        return {
            name + postfix: meter.sum
            for name, meter in self.meters.items()
        }

    def counts(self, postfix='/count'):
        return {
            name + postfix: meter.count
            for name, meter in self.meters.items()
        }


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format)
