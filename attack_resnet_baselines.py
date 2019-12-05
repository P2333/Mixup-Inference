import Torture
import random
import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import MI
from Torture.Models import resnet_3layer as resnet
from Torture.Models import resnet as resnet_imagenet
from MI import pgd
from PIL import Image
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

flags.DEFINE_boolean("-targeted", "--targeted", default=False)

flags.DEFINE_argument("-num_sample", type=int, default=30)

flags.DEFINE_argument("-nbiter", type=int, default=10)

flags.DEFINE_argument("-baseline", default="gaussian")

flags.DEFINE_argument("-sigma", type=float, default=0.05)

flags.DEFINE_argument("-Xielower", type=int, default=20)
flags.DEFINE_argument("-Xieupper", type=int, default=28)

flags.DEFINE_argument("-Guolower", type=int, default=20)
flags.DEFINE_argument("-Guoupper", type=int, default=28)

flags.DEFINE_argument("-Raffprobability", type=float, default=0.2)

flags.DEFINE_argument("-Rotation", type=int, default=10)

flags.DEFINE_argument("-Kurakin", type=float, default=0.1)

flags.DEFINE_argument("-JPEGquality", type=int, default=75)

flags.DEFINE_argument("-eps", type=float, default=8.)

flags.DEFINE_argument("-numtest", type=int, default=50000)

FLAGS = flags.FLAGS

num_test = FLAGS.numtest
#num of times to sample for MI
num_sample = FLAGS.num_sample

logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, FLAGS, config)
logger.info("build dataloader")


def onehot(ind, num_cla):
    vector = np.zeros([num_cla])
    vector[ind] = 1
    return vector.astype(np.float32)

eps_ = 2.
img_size = 32
clip_min, clip_max = -1., 1.
if FLAGS.data.lower() in ["cifar10"]:
    train_trans, test_trans = Torture.Models.transforms.cifar_transform()
    trainset = torchvision.datasets.CIFAR10(
        root='/home/LargeData/cifar/',
        train=False,
        download=True,
        transform=train_trans,
        target_transform=lambda x: onehot(x, 10))
    testset = torchvision.datasets.CIFAR10(
        root='/home/LargeData/cifar/',
        train=False,
        download=True,
        transform=test_trans,
        target_transform=lambda x: onehot(x, 10))
    num_classes = 10
    model_set = resnet
    num_worker = 4
    num_s = 10
elif FLAGS.data.lower() in ["cifar100"]:
    train_trans, test_trans = Torture.Models.transforms.cifar_transform()
    trainset = torchvision.datasets.CIFAR100(
        root='/home/LargeData/cifar/',
        train=False,
        download=True,
        transform=train_trans,
        target_transform=lambda x: onehot(x, 100))
    testset = torchvision.datasets.CIFAR100(
        root='/home/LargeData/cifar/',
        train=False,
        download=True,
        transform=test_trans,
        target_transform=lambda x: onehot(x, 100))
    num_classes = 100
    model_set = resnet
    num_worker = 4
    num_s = 100
elif FLAGS.data.lower() in ["imagenet"]:
    train_trans, test_trans = Torture.Models.transforms.imagenet_transform()
    trainset = torchvision.datasets.ImageNet(
        root='/home/LargeData/ImageNet/',
        split='train',
        download=True,
        transform=train_trans,
        target_transform=lambda x: onehot(x, 1000))
    testset = torchvision.datasets.ImageNet(
        root='/home/LargeData/ImageNet/',
        split='val',
        download=True,
        transform=test_trans,
        target_transform=lambda x: onehot(x, 1000))
    model_set = resnet_imagenet
    num_classes = 1000
    num_worker = 64
    num_s = 1000
    clip_min, clip_max = -2.118, 2.64
    eps_ = 1.
    img_size = 224 # for resnet-50

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

if FLAGS.model.lower() in model_set.model_dict:
    CLASSIFIER = model_set.model_dict[FLAGS.model.lower()]
else:
    raise ValueError("unknown model name")
classifier = CLASSIFIER(num_classes=num_classes)

if FLAGS.data.lower() in ["imagenet"]:
    classifier = nn.DataParallel(classifier)

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
    "eps": FLAGS.eps / 255. * (clip_max - clip_min),
    "eps_iter": eps_ / 255. * (clip_max - clip_min),
    "nb_iter": FLAGS.nbiter,
    "norm": np.inf,
    "clip_min": clip_min,
    "clip_max": clip_max,
    "loss_fn": None,
}

logger.info("Start Attack")

classifier.eval()

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
    img_batch_cpu, label_batch_cpu = data_batch

    # Craft random y_target that not equal to the true labels
    if FLAGS.targeted is True:
        label_target = torch.zeros(label_batch_cpu.shape[0], dtype=torch.int64)
        for j in range(label_batch_cpu.shape[0]):
            l = np.random.randint(num_classes)
            _, j_index = torch.max(label_batch_cpu[j], -1)
            while l == j_index.numpy():
                l = np.random.randint(num_classes)
            label_target[j] = l
        label_target = label_target.to(device)
        pgd_kwargs.update({'y': label_target})

    img_batch, label_batch = img_batch_cpu.to(device), label_batch_cpu.to(
        device)
    adv_x = pgd.projected_gradient_descent(classifier, img_batch, **pgd_kwargs)

    pred_cle_mixup_all = 0
    pred_adv_mixup_all = 0

    # CLEAN
    pred_cle = classifier(img_batch)
    cle_con, predicted_cle = torch.max(soft_max(pred_cle.data), 1)
    predicted_cle = predicted_cle.cpu().numpy()[0]

    #ADVERSARIAL
    pred_adv = classifier(adv_x)
    adv_con, predicted_adv = torch.max(soft_max(pred_adv.data), 1)
    predicted_adv = predicted_adv.cpu().numpy()[0]

    if FLAGS.baseline == 'gaussian':
        sampler_gaussian = torch.distributions.normal.Normal(0, FLAGS.sigma)
        for k in range(num_sample):
            # CLEAN
            noise = sampler_gaussian.sample(sample_shape=img_batch.size())
            mixup_img_cle = noise.to(device) + img_batch
            pred_cle_mixup = classifier(mixup_img_cle)
            pred_cle_mixup_all = pred_cle_mixup_all + soft_max(
                pred_cle_mixup.data)

            #ADVERSARIAL
            noise = sampler_gaussian.sample(sample_shape=img_batch.size())
            mixup_img_adv = noise.to(device) + adv_x
            pred_adv_mixup = classifier(mixup_img_adv)
            pred_adv_mixup_all = pred_adv_mixup_all + soft_max(
                pred_adv_mixup.data)

    elif FLAGS.baseline == 'uniform':
        sampler_gaussian = torch.distributions.uniform.Uniform(
            -FLAGS.sigma, FLAGS.sigma)
        for k in range(num_sample):
            # CLEAN
            noise = sampler_gaussian.sample(sample_shape=img_batch.size())
            mixup_img_cle = noise.to(device) + img_batch
            pred_cle_mixup = classifier(mixup_img_cle)
            pred_cle_mixup_all = pred_cle_mixup_all + soft_max(
                pred_cle_mixup.data)

            #ADVERSARIAL
            noise = sampler_gaussian.sample(sample_shape=img_batch.size())
            mixup_img_adv = noise.to(device) + adv_x
            pred_adv_mixup = classifier(mixup_img_adv)
            pred_adv_mixup_all = pred_adv_mixup_all + soft_max(
                pred_adv_mixup.data)

    elif FLAGS.baseline == 'Rotation':
        for k in range(num_sample):
            #barrage of different transformation with probability p
            T = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomRotation(FLAGS.Rotation),
                torchvision.transforms.ToTensor()
            ])

            # CLEAN
            processed_input_cle = T((img_batch_cpu[0] - clip_min) / (clip_max - clip_min))
            processed_input_cle = (processed_input_cle.unsqueeze(0) * (clip_max - clip_min)) + clip_min
            pred_cle_mixup = classifier(processed_input_cle.to(device))
            pred_cle_mixup_all = pred_cle_mixup_all + soft_max(
                pred_cle_mixup.data)

            #ADVERSARIAL
            processed_input_adv = T((adv_x.cpu()[0] - clip_min) / (clip_max - clip_min))
            processed_input_adv = (processed_input_adv.unsqueeze(0) * (clip_max - clip_min)) + clip_min
            pred_adv_mixup = classifier(processed_input_adv.to(device))
            pred_adv_mixup_all = pred_adv_mixup_all + soft_max(
                pred_adv_mixup.data)
            
    elif FLAGS.baseline == 'Xie':
        for k in range(num_sample):
            #random crop and random padding
            size = random.randint(FLAGS.Xielower, FLAGS.Xieupper)
            a = random.randint(0, img_size - size)  #left padding
            b = random.randint(0, img_size - size)  #top padding
            right_a = img_size - size - a
            bottom_b = img_size - size - b

            T = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomCrop(size),
                torchvision.transforms.Pad((a, b, right_a, bottom_b),
                                           fill=0,
                                           padding_mode='constant'),
                torchvision.transforms.ToTensor()
            ])

            # CLEAN
            processed_input_cle = T((img_batch_cpu[0] - clip_min) / (clip_max - clip_min))
            processed_input_cle = (processed_input_cle.unsqueeze(0) * (clip_max - clip_min)) + clip_min
            pred_cle_mixup = classifier(processed_input_cle.to(device))
            pred_cle_mixup_all = pred_cle_mixup_all + soft_max(
                pred_cle_mixup.data)

            #ADVERSARIAL
            processed_input_adv = T((adv_x.cpu()[0] - clip_min) / (clip_max - clip_min))
            processed_input_adv = (processed_input_adv.unsqueeze(0) * (clip_max - clip_min)) + clip_min
            pred_adv_mixup = classifier(processed_input_adv.to(device))
            pred_adv_mixup_all = pred_adv_mixup_all + soft_max(
                pred_adv_mixup.data)

    elif FLAGS.baseline == 'Guo':
        for k in range(num_sample):
            #random crop and random padding
            size = random.randint(FLAGS.Guolower, FLAGS.Guoupper)

            T = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomCrop(size),
                torchvision.transforms.Resize(img_size, interpolation=2),
                torchvision.transforms.ToTensor()
            ])

            # CLEAN
            processed_input_cle = T((img_batch_cpu[0] - clip_min) / (clip_max - clip_min))
            processed_input_cle = (processed_input_cle.unsqueeze(0) * (clip_max - clip_min)) + clip_min
            pred_cle_mixup = classifier(processed_input_cle.to(device))
            pred_cle_mixup_all = pred_cle_mixup_all + soft_max(
                pred_cle_mixup.data)

            #ADVERSARIAL
            processed_input_adv = T((adv_x.cpu()[0] - clip_min) / (clip_max - clip_min))
            processed_input_adv = (processed_input_adv.unsqueeze(0) * (clip_max - clip_min)) + clip_min
            pred_adv_mixup = classifier(processed_input_adv.to(device))
            pred_adv_mixup_all = pred_adv_mixup_all + soft_max(
                pred_adv_mixup.data) 

    pred_cle_mixup_all = pred_cle_mixup_all / num_sample
    pred_adv_mixup_all = pred_adv_mixup_all / num_sample

    cle_con_mixup, _ = torch.max(pred_cle_mixup_all, 1)
    adv_con_mixup, _ = torch.max(pred_adv_mixup_all, 1)

    if accu(pred_cle, label_batch) == 1:
        #print('Cle Confidence', cle_con)
        #print('Cle_mixup Confidence', cle_con_mixup)
        #print('Adv Confidence', adv_con)
        #print('Adv_mixup Confidence', adv_con_mixup)
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
