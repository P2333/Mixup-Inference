import Torture

import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
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

flags.DEFINE_boolean("-targeted", "--targeted", default=False)

flags.DEFINE_argument("-nbiter", type=int, default=10)

flags.DEFINE_argument("-lamdaPL", type=float, default=0.5)

flags.DEFINE_argument("-lamdaOL", type=float, default=0.5)

flags.DEFINE_argument("-threshold", type=float, default=0.2)
flags.DEFINE_boolean("-use_cifar100_asSamplePool", default=False)

flags.DEFINE_argument("-num_sample_MIOL", type=int, default=15)

flags.DEFINE_argument("-num_sample_MIPL", type=int, default=5)

flags.DEFINE_boolean("-combine_with_Raff", default=False)

flags.DEFINE_boolean("-combine_with_JPEG", default=False)

flags.DEFINE_boolean("-combine_with_Grayscale", default=False)

flags.DEFINE_argument("-Raffprobability",type=float,default=0.2)

flags.DEFINE_argument("-JPEGquality",type=int,default=75)

flags.DEFINE_argument("-eps",type=float,default=8.)


FLAGS = flags.FLAGS

logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, FLAGS, config)
logger.info("build dataloader")

num_test = 1000
num_pool = 10000
#num of times to sample for MI
num_sample_MIPL = FLAGS.num_sample_MIPL
num_sample_MIOL = FLAGS.num_sample_MIOL


def onehot(ind):
    vector = np.zeros([num_classes])
    vector[ind] = 1
    return vector.astype(np.float32)


def onehot100(ind):
    vector = np.zeros([100])
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

############################################
##  Try to use CIFAR-100 example pool in CIFAR-10  ##
trainset_CIFAR100 = torchvision.datasets.CIFAR100(
    root='/home/LargeData/cifar/',
    train=False,
    download=True,
    transform=train_trans,
    target_transform=onehot100)

dataloader_train_CIFAR100 = torch.utils.data.DataLoader(
    trainset_CIFAR100,
    # batch_size=FLAGS.batch_size,
    batch_size=1,
    shuffle=True,
    num_workers=2)

############################################

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

# Whether use CIFAR-100 as the sample pool for CIFAR-10
# if FLAGS.use_cifar100_asSamplePool == True:
#     DL = dataloader_train_CIFAR100
#     num_s = 100
# else:
#     DL = dataloader_train
#     num_s = 10
DL = dataloader_train
num_s=num_classes

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

logger.info("Start Attack")

classifier.eval()

# Construct mixup_pool, mixup_pool[i] contains the subset of training images of label i, 2019/7/26
mixup_pool_PL = {}
mixup_pool_OL = {}

for i in range(num_classes):
    mixup_pool_PL.update({i: []})
for i in range(num_s):
    mixup_pool_OL.update({i: []})

for i, data_batch in enumerate(dataloader_train):
    img_batch, label_batch = data_batch
    img_batch = img_batch.to(device)
    _, label_ind = torch.max(label_batch.data, 1)
    mixup_pool_PL[label_ind.numpy()[0]].append(img_batch)
    if i >= (num_pool - 1):
        break
print('Finish constructing mixup_pool_PL')

for i, data_batch in enumerate(DL):
    img_batch, label_batch = data_batch
    img_batch = img_batch.to(device)
    _, label_ind = torch.max(label_batch.data, 1)
    mixup_pool_OL[label_ind.numpy()[0]].append(img_batch)
    if i >= (num_pool - 1):
        break
print('Finish constructing mixup_pool_OL')

accu_adv, accu_clean = [], []
accu_adv_mixup, accu_clean_mixup = [], []

cle_con_all, cle_con_mixup_all = [], []
adv_con_all, adv_con_mixup_all = [], []

soft_max = nn.Softmax(dim=-1)
classifier.load_state_dict(torch.load(FLAGS.oldmodel))
classifier.eval()

if FLAGS.targeted == True:
    pgd_kwargs.update({'targeted': True})

for i, data_batch in enumerate(dataloader_test):
    # get the inputs; data is a list of [inputs, labels]
    img_batch, label_batch = data_batch

    # Craft random y_target that not equal to the true labels
    if FLAGS.targeted == True:
        label_target = torch.zeros(label_batch.shape[0], dtype=torch.int64)
        for j in range(label_batch.shape[0]):
            l = np.random.randint(num_classes)
            _, j_index = torch.max(label_batch[j], -1)
            while l == j_index.numpy():
                l = np.random.randint(num_classes)
            label_target[j] = l
        label_target = label_target.to(device)
        pgd_kwargs.update({'y': label_target})

    img_batch, label_batch = img_batch.to(device), label_batch.to(device)
    adv_x = pgd.projected_gradient_descent(classifier, img_batch, **pgd_kwargs)

    pred_cle_mixup_all = 0
    pred_adv_mixup_all = 0
    pred_cle_mixup_all_PL = 0
    pred_adv_mixup_all_PL = 0
    pred_cle_mixup_all_OL = 0
    pred_adv_mixup_all_OL = 0

    # CLEAN
    pred_cle = classifier(img_batch)
    cle_con, predicted_cle = torch.max(soft_max(pred_cle.data), 1)
    predicted_cle = predicted_cle.cpu().numpy()[0]

    # ADVERSARIAL
    pred_adv = classifier(adv_x)
    adv_con, predicted_adv = torch.max(soft_max(pred_adv.data), 1)
    predicted_adv = predicted_adv.cpu().numpy()[0]

    # perform MI-PL
    for k in range(num_sample_MIPL):
        # CLEAN
        len_cle = np.random.randint(len(mixup_pool_PL[predicted_cle]))
        mixup_img_cle = (1 - FLAGS.lamdaPL) * mixup_pool_PL[predicted_cle][
            len_cle] + FLAGS.lamdaPL * img_batch
        pred_cle_mixup = classifier(mixup_img_cle)
        pred_cle_mixup_all_PL = pred_cle_mixup_all_PL + soft_max(
            pred_cle_mixup.data)

        # ADVERSARIAL
        len_adv = np.random.randint(len(mixup_pool_PL[predicted_adv]))
        mixup_img_adv = (1 - FLAGS.lamdaPL) * mixup_pool_PL[predicted_adv][
            len_adv] + FLAGS.lamdaPL * adv_x
        pred_adv_mixup = classifier(mixup_img_adv)
        pred_adv_mixup_all_PL = pred_adv_mixup_all_PL + soft_max(
            pred_adv_mixup.data)

    pred_cle_mixup_all_PL = pred_cle_mixup_all_PL / num_sample_MIPL
    pred_adv_mixup_all_PL = pred_adv_mixup_all_PL / num_sample_MIPL

    # perform MI-OL
    for k in range(num_sample_MIOL):
        # CLEAN
        xs_cle_label = np.random.randint(num_s)
        while xs_cle_label == predicted_cle:
            xs_cle_label = np.random.randint(num_s)
        xs_cle_index = np.random.randint(len(mixup_pool_OL[xs_cle_label]))
        mixup_img_cle = (1 - FLAGS.lamdaOL) * mixup_pool_OL[xs_cle_label][xs_cle_index] + FLAGS.lamdaOL * img_batch


        #ADVERSARIAL
        xs_adv_label = np.random.randint(num_s)
        while xs_adv_label == predicted_adv:
            xs_adv_label = np.random.randint(num_s)
        xs_adv_index = np.random.randint(len(mixup_pool_OL[xs_adv_label]))
        mixup_img_adv = (1 - FLAGS.lamdaOL) * mixup_pool_OL[xs_adv_label][xs_adv_index] + FLAGS.lamdaOL * adv_x


        if FLAGS.combine_with_Raff == True:
            T = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomApply([torchvision.transforms.Grayscale(num_output_channels=3)], p=FLAGS.Raffprobability),
                torchvision.transforms.RandomHorizontalFlip(p=FLAGS.Raffprobability),
                torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=FLAGS.Raffprobability, interpolation=3),
                torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(10)], p=FLAGS.Raffprobability),
                torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)], p=FLAGS.Raffprobability),
                torchvision.transforms.ToTensor()
                ])

            processed_input_cle = T((mixup_img_cle.cpu()[0]+1.)/2.)
            processed_input_cle = (processed_input_cle.unsqueeze(0) * 2.) - 1.
            mixup_img_cle = processed_input_cle.to(device)

            processed_input_adv = T((mixup_img_adv.cpu()[0]+1.)/2.)
            processed_input_adv = (processed_input_adv.unsqueeze(0) * 2.) - 1.
            mixup_img_adv = processed_input_adv.to(device)

        elif FLAGS.combine_with_Grayscale == True:
            T = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomApply([torchvision.transforms.Grayscale(num_output_channels=3)], p=0.5),
                torchvision.transforms.ToTensor()
                ])

            processed_input_cle = T((mixup_img_cle.cpu()[0]+1.)/2.)
            processed_input_cle = (processed_input_cle.unsqueeze(0) * 2.) - 1.
            mixup_img_cle = processed_input_cle.to(device)

            processed_input_adv = T((mixup_img_adv.cpu()[0]+1.)/2.)
            processed_input_adv = (processed_input_adv.unsqueeze(0) * 2.) - 1.
            mixup_img_adv = processed_input_adv.to(device)

        elif FLAGS.combine_with_JPEG == True:
            T1 = torchvision.transforms.ToPILImage()
            T2 = torchvision.transforms.ToTensor()
            filename = 'buffer/buffer_'+str(FLAGS.JPEGquality)+'_MIOL_'+str(FLAGS.lamdaOL)+str(FLAGS.targeted)+str(FLAGS.nbiter)+'.jpg'
            # CLEAN
            processed_input_cle = T1((mixup_img_cle.cpu()[0]+1.)/2.)
            processed_input_cle.save(filename, "JPEG", quality=FLAGS.JPEGquality)
            processed_input_cle = Image.open(filename)
            processed_input_cle = T2(processed_input_cle)
            processed_input_cle = (processed_input_cle.unsqueeze(0) * 2.) - 1.
            mixup_img_cle = processed_input_cle.to(device)

            #ADVERSARIAL
            processed_input_adv = T1((mixup_img_adv.cpu()[0]+1.)/2.)
            processed_input_adv.save(filename, "JPEG", quality=FLAGS.JPEGquality)
            processed_input_adv = Image.open(filename)
            processed_input_adv = T2(processed_input_adv)
            processed_input_adv = (processed_input_adv.unsqueeze(0) * 2.) - 1.
            mixup_img_adv = processed_input_adv.to(device)


        pred_cle_mixup = classifier(mixup_img_cle)
        pred_cle_mixup_all_OL = pred_cle_mixup_all_OL + soft_max(pred_cle_mixup.data)

        pred_adv_mixup = classifier(mixup_img_adv)
        pred_adv_mixup_all_OL = pred_adv_mixup_all_OL + soft_max(pred_adv_mixup.data)


    pred_cle_mixup_all_OL = pred_cle_mixup_all_OL / num_sample_MIOL
    pred_adv_mixup_all_OL = pred_adv_mixup_all_OL / num_sample_MIOL

    # Choose MI-PL or MI-OL
    cle_con_mixup_PL, _ = torch.max(pred_cle_mixup_all_PL, 1)
    adv_con_mixup_PL, _ = torch.max(pred_adv_mixup_all_PL, 1)

    gap_cle = cle_con.type(torch.float).item() - cle_con_mixup_PL.type(
        torch.float).item()
    gap_adv = adv_con.type(torch.float).item() - adv_con_mixup_PL.type(
        torch.float).item()

    if gap_cle < FLAGS.threshold:
        pred_cle_mixup_all = pred_cle_mixup_all_PL
    else:
        pred_cle_mixup_all = pred_cle_mixup_all_OL

    if gap_adv < FLAGS.threshold:
        pred_adv_mixup_all = pred_adv_mixup_all_PL
    else:
        pred_adv_mixup_all = pred_adv_mixup_all_OL

    accu_adv.append(accu(pred_adv, label_batch))
    accu_clean.append(accu(pred_cle, label_batch))
    accu_adv_mixup.append(accu(pred_adv_mixup_all, label_batch))
    accu_clean_mixup.append(accu(pred_cle_mixup_all, label_batch))

    if i % 100 == 0:
        print(i)
    if i >= (num_test - 1):
        break

logger.info("Clean ACCU:{}".format(np.mean(accu_clean)))
logger.info("Adver ACCU:{}".format(np.mean(accu_adv)))
logger.info("Clean_mixup ACCU:{}".format(np.mean(accu_clean_mixup)))
logger.info("Adver_mixup ACCU:{}".format(np.mean(accu_adv_mixup)))

logger.info('Finished Attacking')