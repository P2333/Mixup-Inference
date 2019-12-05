import os
import shutil
import time
import socket

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import Utils
from Utils.checkpoints import build_logger
from Utils.checkpoints import plot_image, save_context
from Utils import flags

import Torture
from Torture.Models import resnet_3layer as resnet
import torchvision

from MI import pgd

EVALUATE_EPOCH = 1
SAVE_EPOCH = 10
EPOCH_TOTAL = 200
HYPERPARAMETERS = None
DEFAULT_RESULTS_FOLDER_ARGUMENT = "Not Valid"
DEFAULT_RESULTS_FOLDER = "./results/"
FILES_TO_BE_SAVED = ["./", "./Torture", "./Torture/Models", "./Utils"]
KEY_ARGUMENTS = ["batch_size", "model", "data", "adv_ratio"]
config = {
    "DEFAULT_RESULTS_FOLDER": DEFAULT_RESULTS_FOLDER,
    "FILES_TO_BE_SAVED": FILES_TO_BE_SAVED,
    "KEY_ARGUMENTS": KEY_ARGUMENTS
}

flags.DEFINE_argument("-gpu", "--gpu", default="-1")
flags.DEFINE_argument("--results-folder",
                      default=DEFAULT_RESULTS_FOLDER_ARGUMENT)
flags.DEFINE_argument("-k", "-key", "--key", default="")
flags.DEFINE_argument("-data", "--data", default="Caltech101")
flags.DEFINE_boolean("-o", "--overwrite-results", default=False)
flags.DEFINE_argument("-bs",
                      "-batch_size",
                      "--batch_size",
                      type=int,
                      default=50)
flags.DEFINE_argument("-nw",
                      "-num_workers",
                      "--num_workers",
                      type=int,
                      default=64)
flags.DEFINE_argument("-ar",
                      "-adv_ratio",
                      "--adv_ratio",
                      type=float,
                      default=0.)
flags.DEFINE_argument("-model", "--model", default="resnet18")
FLAGS = flags.FLAGS

logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, FLAGS, config)
logger.info("build dataloader")

with open("TotalList.txt", "a") as f:
    f.write(socket.gethostname() + ":" + FLAGS.results_folder + "\n")
# Figure finished


def onehot(ind):
    vector = np.zeros([num_classes])
    vector[ind] = 1
    return vector.astype(np.float32)


logger.info("build dataloader")

if FLAGS.data.lower() in ["cifar10"]:
    train_trans, test_trans = Torture.Models.transforms.cifar_transform()
    trainset = torchvision.datasets.CIFAR10(root='/home/LargeData/cifar/',
                                            train=True,
                                            download=True,
                                            transform=train_trans)
    testset = torchvision.datasets.CIFAR10(root='/home/LargeData/cifar/',
                                           train=False,
                                           download=True,
                                           transform=test_trans)
    num_classes = 10
elif FLAGS.data.lower() in ["cifar100"]:
    train_trans, test_trans = Torture.Models.transforms.cifar_transform()
    trainset = torchvision.datasets.CIFAR100(root='/home/LargeData/cifar/',
                                             train=True,
                                             download=True,
                                             transform=train_trans)
    testset = torchvision.datasets.CIFAR100(root='/home/LargeData/cifar/',
                                            train=False,
                                            download=True,
                                            transform=test_trans)
    num_classes = 100

dataloader_train = torch.utils.data.DataLoader(trainset,
                                               batch_size=FLAGS.batch_size,
                                               shuffle=True,
                                               num_workers=2)

dataloader_test = torch.utils.data.DataLoader(testset,
                                              batch_size=FLAGS.batch_size,
                                              shuffle=False,
                                              num_workers=2)

if FLAGS.model.lower() in resnet.model_dict:
    CLASSIFIER = resnet.model_dict[FLAGS.model.lower()]
else:
    raise ValueError("unknown model name")
classifier = CLASSIFIER(num_classes=num_classes)

device = torch.device("cuda:0")
classifier = classifier.to(device)
# classifier = nn.DataParallel(classifier)


def anneal_lr(epoch):
    if epoch < 100:
        return 1.
    elif epoch < 150:
        return 0.1
    else:
        return 0.01


pgd_kwargs = {
    "eps": 16. / 255.,
    "eps_iter": 4. / 255.,
    "nb_iter": 10,
    "norm": np.inf,
    "clip_min": -1,
    "clip_max": 1,
    "loss_fn": None,
}

criterion = nn.CrossEntropyLoss()
criterion_adv = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, [anneal_lr])

for epoch in range(EPOCH_TOTAL):  # loop over the dataset multiple times
    logger.info("Start Epoch {}".format(epoch))
    running_loss = 0.0
    lr_scheduler.step()
    classifier.train()
    for i, data_batch in enumerate(dataloader_train):
        # get the inputs; data is a list of [inputs, labels]
        img_batch, label_batch = data_batch
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        loss = 0.
        if FLAGS.adv_ratio > 0.:
            classifier.eval()
            adv_x = pgd.projected_gradient_descent(classifier, img_batch,
                                                   **pgd_kwargs)
            classifier.train()
            optimizer.zero_grad()
            output_batch_adv = classifier(adv_x)
            loss += criterion_adv(output_batch_adv,
                                  label_batch) * FLAGS.adv_ratio
        else:
            optimizer.zero_grad()

        if FLAGS.adv_ratio < 1.:
            output_batch = classifier(img_batch)
            loss += criterion(output_batch,
                              label_batch) * (1. - FLAGS.adv_ratio)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    logger.info('[%d] train loss: %.3f' % (epoch + 1, running_loss / i))

    if epoch % EVALUATE_EPOCH == 0:
        running_loss, correct, total = 0.0, 0.0, 0.0
        classifier.eval()
        for i, data_batch in enumerate(dataloader_test):
            # get the inputs; data is a list of [inputs, labels]
            img_batch, label_batch = data_batch
            img_batch, label_batch = img_batch.to(device), label_batch.to(
                device)
            output_batch = classifier(img_batch)
            loss = criterion(output_batch, label_batch)
            running_loss += loss.item()

            _, predicted = torch.max(output_batch.data, 1)
            correct += (predicted == label_batch).sum().item()
            total += label_batch.size(0)
        logger.info('[%d] test loss: %.3f, accuracy: %.3f' %
                    (epoch + 1, running_loss / i, correct / total))

    if epoch % SAVE_EPOCH == 0 or epoch == EPOCH_TOTAL - 1:
        torch.save(classifier.state_dict(),
                   os.path.join(MODELS_FOLDER, "eopch{}.ckpt".format(epoch)))

logger.info('Finished Training')
