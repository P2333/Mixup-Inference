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
from Torture.loss_function import soft_cross_entropy
from Torture.Models import resnet_3layer as resnet
from Torture.Models import resnet as resnet_imagenet
import torchvision

from MI import pgd

EVALUATE_EPOCH = 1
SAVE_EPOCH = 10
EPOCH_TOTAL = 200
HYPERPARAMETERS = None
DEFAULT_RESULTS_FOLDER_ARGUMENT = "Not Valid"
DEFAULT_RESULTS_FOLDER = "./results/"
FILES_TO_BE_SAVED = ["./", "./Torture", "./Torture/Models", "./Utils"]
KEY_ARGUMENTS = ["batch_size", "model", "data", "adv_ratio", "mixup_alpha"]
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
flags.DEFINE_argument("-lr",
                      "-learning_rate",
                      "--learning_rate",
                      type=float,
                      default=0.01)
flags.DEFINE_argument("-nw",
                      "-num_workers",
                      "--num_workers",
                      type=int,
                      default=64)
flags.DEFINE_argument("-ar",
                      "-adv_ratio",
                      "--adv_ratio",
                      type=float,
                      default=10)
flags.DEFINE_argument("-ma",
                      "-mixup_alpha",
                      "--mixup_alpha",
                      type=float,
                      default=0.)
flags.DEFINE_argument("-model", "--model", default="resnet18")
flags.DEFINE_argument("-save_epoch",
                      "--save_epoch",
                      type=int,
                      default=SAVE_EPOCH)
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
                                            transform=train_trans,
                                            target_transform=onehot)
    testset = torchvision.datasets.CIFAR10(root='/home/LargeData/cifar/',
                                           train=False,
                                           download=True,
                                           transform=test_trans,
                                           target_transform=onehot)
    model_set = resnet
    num_classes = 10
    num_worker = 2
elif FLAGS.data.lower() in ["cifar100"]:
    train_trans, test_trans = Torture.Models.transforms.cifar_transform()
    trainset = torchvision.datasets.CIFAR100(root='/home/LargeData/cifar/',
                                             train=True,
                                             download=True,
                                             transform=train_trans,
                                             target_transform=onehot)
    testset = torchvision.datasets.CIFAR100(root='/home/LargeData/cifar/',
                                            train=False,
                                            download=True,
                                            transform=test_trans,
                                            target_transform=onehot)
    model_set = resnet
    num_classes = 100
    num_worker = 2
elif FLAGS.data.lower() in ["imagenet"]:
    train_trans, test_trans = Torture.Models.transforms.imagenet_transform()
    trainset = torchvision.datasets.ImageNet(root='/home/LargeData/ImageNet/',
                                             split='train',
                                             download=True,
                                             transform=train_trans,
                                             target_transform=onehot)
    testset = torchvision.datasets.ImageNet(root='/home/LargeData/ImageNet/',
                                            split='val',
                                            download=True,
                                            transform=test_trans,
                                            target_transform=onehot)
    model_set = resnet_imagenet
    num_classes = 1000
    num_worker = 64

dataloader_train = torch.utils.data.DataLoader(trainset,
                                               batch_size=FLAGS.batch_size,
                                               shuffle=True,
                                               num_workers=num_worker,
                                               drop_last=True)

dataloader_test = torch.utils.data.DataLoader(testset,
                                              batch_size=FLAGS.batch_size,
                                              shuffle=False,
                                              num_workers=num_worker,
                                              drop_last=True)

if FLAGS.model.lower() in model_set.model_dict:
    CLASSIFIER = model_set.model_dict[FLAGS.model.lower()]
else:
    raise ValueError("unknown model name")

if FLAGS.data.lower() in ["imagenet"]:
    classifier = CLASSIFIER(num_classes=num_classes, pretrained=True)
    classifier = nn.DataParallel(classifier)
else:
    classifier = CLASSIFIER(num_classes=num_classes)

device = torch.device("cuda:0")
classifier = classifier.to(device)


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


def shuffle_minibatch(inputs, targets):
    if FLAGS.mixup_alpha == 0.:
        return inputs, targets
    mixup_alpha = FLAGS.mixup_alpha

    batch_size = inputs.shape[0]
    rp1 = torch.randperm(batch_size)
    inputs1 = inputs[rp1]
    targets1 = targets[rp1]

    rp2 = torch.randperm(batch_size)
    inputs2 = inputs[rp2]
    targets2 = targets[rp2]

    ma = np.random.beta(mixup_alpha, mixup_alpha, [batch_size, 1])
    ma_img = ma[:, :, None, None]

    inputs1 = inputs1 * torch.from_numpy(ma_img).to(device).float()
    inputs2 = inputs2 * torch.from_numpy(1 - ma_img).to(device).float()

    targets1 = targets1.float() * torch.from_numpy(ma).to(device).float()
    targets2 = targets2.float() * torch.from_numpy(1 - ma).to(device).float()

    inputs_shuffle = (inputs1 + inputs2).to(device)
    targets_shuffle = (targets1 + targets2).to(device)

    return inputs_shuffle, targets_shuffle


criterion = soft_cross_entropy
optimizer = optim.SGD(classifier.parameters(),
                      lr=FLAGS.learning_rate, momentum=0.9)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, [anneal_lr])

for epoch in range(EPOCH_TOTAL):  # loop over the dataset multiple times
    logger.info("Start Epoch {}".format(epoch))
    running_loss_1, running_loss_2 = 0.0, 0.0
    lr_scheduler.step()

    for i, data_batch in enumerate(dataloader_train):
        # get the inputs; data is a list of [inputs, labels]
        # if i == 19:
        #     break
        img_batch, label_batch = data_batch
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)

        train_img_batch, train_label_batch = [], []
        classifier.eval()
        if FLAGS.adv_ratio > 0.:
            adv_x = pgd.projected_gradient_descent(classifier, img_batch,
                                                   **pgd_kwargs)
            adv_x_s, adv_label_batch_s = shuffle_minibatch(adv_x, label_batch)
            train_img_batch.append(adv_x_s)
            train_label_batch.append(adv_label_batch_s)

        if FLAGS.adv_ratio < 1.:
            img_batch_s, label_batch_s = shuffle_minibatch(
                img_batch, label_batch)
            train_img_batch.append(img_batch_s)
            train_label_batch.append(label_batch_s)

        train_img_batch = torch.cat(train_img_batch, dim=0)
        classifier.train()
        output_batch = classifier(train_img_batch)

        if len(train_label_batch) == 1:
            loss = criterion(output_batch, train_label_batch[0])
            if FLAGS.adv_ratio == 0.:
                running_loss_2 += loss.item()
            else:
                running_loss_1 += loss.item()
        else:
            output_batch1 = output_batch[:FLAGS.batch_size]
            output_batch2 = output_batch[FLAGS.batch_size:]
            loss1 = criterion(output_batch1,
                              train_label_batch[0]) * FLAGS.adv_ratio
            loss2 = criterion(output_batch2,
                              train_label_batch[1]) * (1. - FLAGS.adv_ratio)
            running_loss_1 += loss1.item()
            running_loss_2 += loss2.item()
            loss = loss1 + loss2

        classifier.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info('[%d] train loss: adv: %.3f, clean: %.3f' %
                (epoch + 1, running_loss_1 / i, running_loss_2 / i))

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
            _, label_ind = torch.max(label_batch.data, 1)
            correct += (predicted == label_ind).sum().item()
            total += label_batch.size(0)
        logger.info('[%d] test loss: %.3f, accuracy: %.3f' %
                    (epoch + 1, running_loss / i, correct / total))

    if epoch % FLAGS.save_epoch == 0 or epoch == EPOCH_TOTAL - 1:
        torch.save(classifier.state_dict(),
                   os.path.join(MODELS_FOLDER, "eopch{}.ckpt".format(epoch)))

logger.info('Finished Training')
