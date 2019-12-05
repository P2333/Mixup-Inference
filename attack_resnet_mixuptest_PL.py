import Torture

import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import torchvision
import MI
from Torture.Models import resnet_3layer as resnet
from MI import pgd

from Utils.checkpoints import plot_image, save_context
from Utils import flags

#torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#np.random.seed(12345)

EVALUATE_EPOCH = 1
SAVE_EPOCH = 5
HYPERPARAMETERS = None

DEFAULT_RESULTS_FOLDER_ARGUMENT = "Not Valid"
DEFAULT_RESULTS_FOLDER = "~/results/"
FILES_TO_BE_SAVED = ["./", "./Torture", "./Torture/Models", "./Utils", "MI"]
KEY_ARGUMENTS = ["batch_size", "model", "data"]
config = {
    "DEFAULT_RESULTS_FOLDER": DEFAULT_RESULTS_FOLDER,
    "FILES_TO_BE_SAVED": FILES_TO_BE_SAVED,
    "KEY_ARGUMENTS": KEY_ARGUMENTS
}

flags.DEFINE_argument("-gpu", "--gpu", default="-1")
flags.DEFINE_argument("--results-folder",
                      default=DEFAULT_RESULTS_FOLDER_ARGUMENT)
flags.DEFINE_argument("-k", "-key", "--key", default="")
flags.DEFINE_argument("-data", "--data", default="cifar10")
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
                      default=4)
flags.DEFINE_argument("-oldmodel", "--oldmodel", default=None)
flags.DEFINE_argument("-model", "--model", default="resnet34")
flags.DEFINE_boolean("-adaptive", "--adaptive", default=False)

flags.DEFINE_boolean("-targeted", "--targeted", default=False)

flags.DEFINE_argument("-nbiter", type=int, default=10)

flags.DEFINE_argument("-lamda", type=float, default=0.5)

flags.DEFINE_argument("-eps", type=float, default=8.)

FLAGS = flags.FLAGS

logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, FLAGS, config)
logger.info("build dataloader")

num_pool = 1000
num_test = 1000
# num of times to sample for MI
num_sample = 10


def onehot(ind):
    vector = np.zeros([num_classes])
    vector[ind] = 1
    return vector.astype(np.float32)


if FLAGS.data.lower() in ["cifar10"]:
    train_trans, test_trans = Torture.Models.transforms.cifar_transform()
    trainset = torchvision.datasets.CIFAR10(root='/home/LargeData/cifar/',
                                            train=False,
                                            download=True,
                                            transform=train_trans,
                                            target_transform=onehot)
    testset = torchvision.datasets.CIFAR10(root='/home/LargeData/cifar/',
                                           train=False,
                                           download=True,
                                           transform=test_trans,
                                           target_transform=onehot)
    num_classes = 10
elif FLAGS.data.lower() in ["cifar100"]:
    train_trans, test_trans = Torture.Models.transforms.cifar_transform()
    trainset = torchvision.datasets.CIFAR100(root='/home/LargeData/cifar/',
                                             train=False,
                                             download=True,
                                             transform=train_trans,
                                             target_transform=onehot)
    testset = torchvision.datasets.CIFAR100(root='/home/LargeData/cifar/',
                                            train=False,
                                            download=True,
                                            transform=test_trans,
                                            target_transform=onehot)
    num_classes = 100

dataloader_train = torch.utils.data.DataLoader(
    trainset,
    # batch_size=FLAGS.batch_size,
    batch_size=1,
    shuffle=True,
    num_workers=2)

dataloader_test = torch.utils.data.DataLoader(
    testset,
    # batch_size=FLAGS.batch_size,
    batch_size=1,
    shuffle=False,
    num_workers=2)

if FLAGS.model.lower() in resnet.model_dict:
    CLASSIFIER = resnet.model_dict[FLAGS.model.lower()]
else:
    raise ValueError("unknown model name")
classifier = CLASSIFIER(num_classes=num_classes)

device = torch.device("cuda:0")
classifier = classifier.to(device)

logger.info("Load model from" + FLAGS.oldmodel)
classifier.load_state_dict(torch.load(FLAGS.oldmodel))


def accu(pred, label):
    _, predicted = torch.max(pred.data, 1)
    _, label_ind = torch.max(label.data, 1)
    result = (predicted == label_ind).type(torch.float)
    return result.mean().item()


pgd_kwargs = {
    "eps": FLAGS.eps * 2. / 255.,
    "eps_iter": 4. / 255.,
    "nb_iter": FLAGS.nbiter,
    "norm": np.inf,
    "clip_min": -1,
    "clip_max": 1,
    "loss_fn": None,
}


def gen_adv_crafter(_x, _y=None, **kwargs):
    kwargs.update(pgd_kwargs)
    classifier.eval()
    return MI.pgd.projected_gradient_descent(classifier, _x, y=_y, **kwargs)


def gen_adv_crafter_adaptive(_x, _y, adaptive_pool):
    assert _x.size(0) == 1
    if _y is not None:
        _y = torch.repeat_interleave(_y, num_sample, 0)

    origin_x = (1 - FLAGS.lamda) * adaptive_pool + FLAGS.lamda * _x
    adv_x = gen_adv_crafter(origin_x, _y, adaptive=True)
    perturbation = adv_x - origin_x
    perturbation = torch.mean(perturbation, dim=0, keepdim=True)
    _x = _x + perturbation
    return _x


logger.info("Start Attack")

classifier.eval()

# Construct mixup_pool, mixup_pool[i] contains the subset of training images of label i, 2019/7/26
mixup_pool = {}
for i in range(num_classes):
    mixup_pool.update({i: []})

for i, data_batch in enumerate(dataloader_train):
    # for i, data_batch in enumerate(dataloader_test):
    img_batch, label_batch = data_batch
    img_batch = img_batch.to(device)
    _, label_ind = torch.max(label_batch.data, 1)
    mixup_pool[label_ind.numpy()[0]].append(img_batch)
    if i >= (num_pool - 1):
        break
print('Finish constructing mixup_pool')

accu_adv, accu_clean = [], []
accu_adv_mixup, accu_clean_mixup = [], []

cle_con_all, cle_con_mixup_all = [], []
adv_con_all, adv_con_mixup_all = [], []

soft_max = nn.Softmax(dim=-1)
classifier.load_state_dict(torch.load(FLAGS.oldmodel))
classifier.eval()

if FLAGS.targeted is True:
    pgd_kwargs.update({'targeted': True})

for i, data_batch in enumerate(dataloader_test):
    # get the inputs; data is a list of [inputs, labels]
    img_batch, label_batch = data_batch

    # Craft random y_target that not equal to the true labels
    if FLAGS.targeted is True:
        label_target = torch.zeros(label_batch.shape[0], dtype=torch.int64)
        for j in range(label_batch.shape[0]):
            _l = np.random.randint(num_classes)
            _, j_index = torch.max(label_batch[j], -1)
            while _l == j_index.numpy():
                _l = np.random.randint(num_classes)
            label_target[j] = _l
        label_target = label_target.to(device)
    else:
        label_target = None

    img_batch, label_batch = img_batch.to(device), label_batch.to(device)
    # CLEAN
    pred_cle = classifier(img_batch)
    cle_con, predicted_cle = torch.max(soft_max(pred_cle.data), 1)
    predicted_cle = predicted_cle.cpu().numpy()[0]

    if FLAGS.adaptive is True:
        adaptive_x_pool = []
        for _ in range(num_sample):
            image_ind = np.random.randint(len(mixup_pool[predicted_cle]))
            adaptive_x_pool.append(mixup_pool[predicted_cle][image_ind])
        adaptive_x_pool = torch.cat(adaptive_x_pool, dim=0)
        adv_x = gen_adv_crafter_adaptive(img_batch, label_target,
                                         adaptive_x_pool)
    else:
        adv_x = gen_adv_crafter(img_batch, label_target)

    pred_cle_mixup_all = 0
    pred_adv_mixup_all = 0

    # ADVERSARIAL
    pred_adv = classifier(adv_x)
    adv_con, predicted_adv = torch.max(soft_max(pred_adv.data), 1)
    predicted_adv = predicted_adv.cpu().numpy()[0]

    for k in range(num_sample):
        # CLEAN
        len_cle = np.random.randint(len(mixup_pool[predicted_cle]))
        mixup_img_cle = (1 - FLAGS.lamda) * mixup_pool[predicted_cle][
            len_cle] + FLAGS.lamda * img_batch
        pred_cle_mixup = classifier(mixup_img_cle)
        pred_cle_mixup_all = pred_cle_mixup_all + soft_max(pred_cle_mixup.data)

        # ADVERSARIAL
        len_adv = np.random.randint(len(mixup_pool[predicted_adv]))
        mixup_img_adv = (1 - FLAGS.lamda) * mixup_pool[predicted_adv][
            len_adv] + FLAGS.lamda * adv_x
        pred_adv_mixup = classifier(mixup_img_adv)
        pred_adv_mixup_all = pred_adv_mixup_all + soft_max(pred_adv_mixup.data)

    pred_cle_mixup_all = pred_cle_mixup_all / num_sample
    pred_adv_mixup_all = pred_adv_mixup_all / num_sample

    cle_con_mixup, _ = torch.max(pred_cle_mixup_all, 1)
    adv_con_mixup, _ = torch.max(pred_adv_mixup_all, 1)

    if accu(pred_cle, label_batch) == 1:
        # print('Cle Confidence', cle_con)
        # print('Cle_mixup Confidence', cle_con_mixup)
        # print('Adv Confidence', adv_con)
        # print('Adv_mixup Confidence', adv_con_mixup)
        cle_con_all.append(cle_con.type(torch.float).item())
        cle_con_mixup_all.append(cle_con_mixup.type(torch.float).item())
        adv_con_all.append(adv_con.type(torch.float).item())
        adv_con_mixup_all.append(adv_con_mixup.type(torch.float).item())

    accu_adv.append(accu(pred_adv, label_batch))
    accu_clean.append(accu(pred_cle, label_batch))
    accu_adv_mixup.append(accu(pred_adv_mixup_all, label_batch))
    accu_clean_mixup.append(accu(pred_cle_mixup_all, label_batch))

    if i % 100 == 0:
        print(i)
    if i >= (num_test - 1):
        break

gap_con = np.array(cle_con_all) - np.array(cle_con_mixup_all)
gap_con_adv = np.array(adv_con_all) - np.array(adv_con_mixup_all)

logger.info("Clean ACCU:{}".format(np.mean(accu_clean)))
logger.info("Adver ACCU:{}".format(np.mean(accu_adv)))
logger.info("Clean_mixup ACCU:{}".format(np.mean(accu_clean_mixup)))
logger.info("Adver_mixup ACCU:{}".format(np.mean(accu_adv_mixup)))

logger.info("Clean con:{} std:{}".format(np.mean(cle_con_all),
                                         np.std(cle_con_all)))
logger.info("Clean_mixup con:{} std:{}".format(np.mean(cle_con_mixup_all),
                                               np.std(cle_con_mixup_all)))
logger.info("Clean_gap con:{} std:{}".format(np.mean(gap_con),
                                             np.std(gap_con)))

logger.info("Adver con:{} std:{}".format(np.mean(adv_con_all),
                                         np.std(adv_con_all)))
logger.info("Adver_mixup con:{} std:{}".format(np.mean(adv_con_mixup_all),
                                               np.std(adv_con_mixup_all)))
logger.info("Adver_gap con:{} std:{}".format(np.mean(gap_con_adv),
                                             np.std(gap_con_adv)))

logger.info('Finished Attacking')