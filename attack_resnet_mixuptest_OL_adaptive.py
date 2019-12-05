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
from MI import art
from Torture.Models import resnet_3layer as resnet
from Torture.Models import resnet as resnet_imagenet
from MI import pgd

from Utils.checkpoints import plot_image, save_context
from Utils import flags

# torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# np.random.seed(12345)

EVALUATE_EPOCH = 1
SAVE_EPOCH = 5
HYPERPARAMETERS = None

DEFAULT_RESULTS_FOLDER_ARGUMENT = "Not Valid"
DEFAULT_RESULTS_FOLDER = "./results_test/"
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
flags.DEFINE_argument("-attack", "--attack", default="cleverhans_pgd")
flags.DEFINE_boolean("-adaptive", "--adaptive", default=False)
flags.DEFINE_argument("-adaptive_num", "--adaptive_num", type=int, default=1)

flags.DEFINE_boolean("-targeted", "--targeted", default=False)

flags.DEFINE_argument("-nbiter", type=int, default=10)

flags.DEFINE_argument("-lamda", type=float, default=0.5)

flags.DEFINE_argument("-num_sample", type=int, default=30)

flags.DEFINE_boolean("-use_cifar100_asSamplePool", default=False)

flags.DEFINE_boolean("-combine_with_Raff", default=False)

flags.DEFINE_boolean("-combine_with_JPEG", default=False)

flags.DEFINE_boolean("-combine_with_Grayscale", default=False)

flags.DEFINE_argument("-Raffprobability", type=float, default=0.2)

flags.DEFINE_argument("-JPEGquality", type=int, default=75)

flags.DEFINE_argument("-eps", type=float, default=8.)

FLAGS = flags.FLAGS

logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, FLAGS, config)
logger.info("build dataloader")

num_test = 1000
num_pool = 10000
# num of times to sample for MI
num_sample = FLAGS.num_sample


def onehot(ind, num_cla):
    vector = np.zeros([num_cla])
    vector[ind] = 1
    return vector.astype(np.float32)


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

# ###########################################
# #  Try to use CIFAR-100 example pool in CIFAR-10  ##
trainset_CIFAR100 = torchvision.datasets.CIFAR100(
    root='/home/LargeData/cifar/',
    train=False,
    download=True,
    transform=train_trans,
    target_transform=lambda x: onehot(x, 100))

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
if FLAGS.use_cifar100_asSamplePool is True:
    DL = dataloader_train_CIFAR100
    num_s = 100
else:
    DL = dataloader_train

if FLAGS.model.lower() in model_set.model_dict:
    CLASSIFIER = model_set.model_dict[FLAGS.model.lower()]
else:
    raise ValueError("unknown model name")
classifier = CLASSIFIER(num_classes=num_classes)

device = torch.device("cuda:0")
classifier = classifier.to(device)

logger.info("Load model from" + FLAGS.oldmodel)
classifier.load_state_dict(torch.load(FLAGS.oldmodel))
classifier.eval()
cla_wraper = art.PyTorchClassifier(clip_values=(clip_min, clip_max),
                                   model=classifier,
                                   loss=nn.CrossEntropyLoss(),
                                   optimizer=optim.SGD(classifier.parameters(),
                                                       lr=0.01,
                                                       momentum=0.9),
                                   input_shape=(3, 32, 32),
                                   nb_classes=num_classes)


def accu(pred, label):
    _, predicted = torch.max(pred.data, 1)
    _, label_ind = torch.max(label.data, 1)
    result = (predicted == label_ind).type(torch.float)
    return result.mean().item()


if FLAGS.attack.lower() in ['carlinil2']:
    att_params = {
        'targeted': False,
        'batch_size': 100,
        'confidence': 0.1,
        'learning_rate': 0.01,
        'binary_search_steps': 1,
        'max_iter': 1000,
        'initial_const': 0.01,
    }
    if FLAGS.targeted is True:
        att_params.update({'targeted': True})
    attack_obj = art.carlini.CarliniL2Method(classifier=cla_wraper,
                                             **att_params)

    def _crafter(x, y=None):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if y is not None and isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        adv_x = attack_obj.generate(x, y)
        adv_x = torch.from_numpy(adv_x).to(device)
        return adv_x

    gen_adv_crafter = _crafter
elif FLAGS.attack.lower() in ['pgd']:
    att_params = {
        "eps": FLAGS.eps * 2. / 255.,
        "eps_step": 4. / 255.,
        "max_iter": 10,
        "norm": np.inf,
        "num_random_init": 1,
        "targeted": False,
    }
    if FLAGS.targeted is True:
        att_params.update({'targeted': True})
    attack_obj = art.pgd.ProjectedGradientDescent(classifier=cla_wraper,
                                                  **att_params)

    def _crafter(x, y=None):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if y is not None and isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        adv_x = attack_obj.generate(x, y)
        adv_x = torch.from_numpy(adv_x).to(device)
        return adv_x

    gen_adv_crafter = _crafter
elif FLAGS.attack.lower() in ['cleverhans_pgd']:
    att_params = {
        "eps": FLAGS.eps * 2. / 255.,
        "eps_iter": 4. / 255.,
        "nb_iter": FLAGS.nbiter,
        "norm": np.inf,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "loss_fn": None,
        "targeted": False,
    }

    if FLAGS.targeted is True:
        att_params.update({'targeted': True})

    def _crafter(x, y=None, **kwargs):
        kwargs.update(att_params)
        classifier.eval()
        return MI.pgd.projected_gradient_descent(classifier, x, y=y, **kwargs)

    gen_adv_crafter = _crafter
else:
    raise ValueError("Unknown argument attack: {}".format(FLAGS.attack))


def gen_adv_crafter_adaptive(_x, _y, adaptive_pool):
    assert _x.size(0) == 1
    if _y is not None:
        _y = torch.repeat_interleave(_y, FLAGS.adaptive_num, 0)
    origin_x = (1 - FLAGS.lamda) * adaptive_pool + FLAGS.lamda * _x
    adv_x = gen_adv_crafter(origin_x, _y, adaptive=True)
    perturbation = adv_x - origin_x
    perturbation = torch.mean(perturbation, dim=0, keepdim=True)
    _x = _x + perturbation
    return _x
    # return gen_adv_crafter(_x, _y)


logger.info("Start Attack")

classifier.eval()

# Construct mixup_pool, mixup_pool[i] contains the subset of
# training images of label i, 2019/7/26
mixup_pool = {}
# for i in range(10):
for i in range(num_s):
    mixup_pool.update({i: []})

# for i, data_batch in enumerate(dataloader_train):
# for i, data_batch in enumerate(dataloader_test):
for i, data_batch in enumerate(DL):
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

    img_batch = img_batch.to(device)
    label_batch = label_batch.to(device)

    # CLEAN
    pred_cle = classifier(img_batch)
    cle_con, predicted_cle = torch.max(soft_max(pred_cle.data), 1)
    predicted_cle = predicted_cle.cpu().numpy()[0]

    if FLAGS.adaptive is True:
        adaptive_x_pool = []
        for _ in range(FLAGS.adaptive_num):
            class_ind = np.random.randint(num_s)
            while class_ind == predicted_cle:
                class_ind = np.random.randint(num_s)
            image_ind = np.random.randint(len(mixup_pool[class_ind]))
            adaptive_x_pool.append(mixup_pool[class_ind][image_ind])
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
        xs_cle_label = np.random.randint(num_s)
        while xs_cle_label == predicted_cle:
            xs_cle_label = np.random.randint(num_s)
        xs_cle_index = np.random.randint(len(mixup_pool[xs_cle_label]))
        mixup_img_cle = (1 - FLAGS.lamda) * \
            mixup_pool[xs_cle_label][xs_cle_index] + FLAGS.lamda * img_batch

        # ADVERSARIAL
        xs_adv_label = np.random.randint(num_s)
        while xs_adv_label == predicted_adv:
            xs_adv_label = np.random.randint(num_s)
        xs_adv_index = np.random.randint(len(mixup_pool[xs_adv_label]))
        mixup_img_adv = (1 - FLAGS.lamda) * \
            mixup_pool[xs_adv_label][xs_adv_index] + FLAGS.lamda * adv_x

        if FLAGS.combine_with_Raff is True:
            T = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.Grayscale(num_output_channels=3)],
                    p=FLAGS.Raffprobability),
                torchvision.transforms.RandomHorizontalFlip(
                    p=FLAGS.Raffprobability),
                torchvision.transforms.RandomPerspective(
                    distortion_scale=0.5,
                    p=FLAGS.Raffprobability,
                    interpolation=3),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.RandomRotation(10)],
                    p=FLAGS.Raffprobability),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1)
                ],
                                                   p=FLAGS.Raffprobability),
                torchvision.transforms.ToTensor()
            ])

            processed_input_cle = T((mixup_img_cle.cpu()[0] + 1.) / 2.)
            processed_input_cle = (processed_input_cle.unsqueeze(0) * 2.) - 1.
            mixup_img_cle = processed_input_cle.to(device)

            processed_input_adv = T((mixup_img_adv.cpu()[0] + 1.) / 2.)
            processed_input_adv = (processed_input_adv.unsqueeze(0) * 2.) - 1.
            mixup_img_adv = processed_input_adv.to(device)

        elif FLAGS.combine_with_Grayscale is True:
            T = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.Grayscale(num_output_channels=3)],
                    p=0.5),
                torchvision.transforms.ToTensor()
            ])

            processed_input_cle = T((mixup_img_cle.cpu()[0] + 1.) / 2.)
            processed_input_cle = (processed_input_cle.unsqueeze(0) * 2.) - 1.
            mixup_img_cle = processed_input_cle.to(device)

            processed_input_adv = T((mixup_img_adv.cpu()[0] + 1.) / 2.)
            processed_input_adv = (processed_input_adv.unsqueeze(0) * 2.) - 1.
            mixup_img_adv = processed_input_adv.to(device)

        elif FLAGS.combine_with_JPEG is True:
            T1 = torchvision.transforms.ToPILImage()
            T2 = torchvision.transforms.ToTensor()
            filename = 'buffer/buffer_' + str(
                FLAGS.JPEGquality) + '_MIOL_' + str(FLAGS.lamda) + str(
                    FLAGS.targeted) + str(FLAGS.nbiter) + '.jpg'
            # CLEAN
            processed_input_cle = T1((mixup_img_cle.cpu()[0] + 1.) / 2.)
            processed_input_cle.save(filename,
                                     "JPEG",
                                     quality=FLAGS.JPEGquality)
            processed_input_cle = Image.open(filename)
            processed_input_cle = T2(processed_input_cle)
            processed_input_cle = (processed_input_cle.unsqueeze(0) * 2.) - 1.
            mixup_img_cle = processed_input_cle.to(device)

            # ADVERSARIAL
            processed_input_adv = T1((mixup_img_adv.cpu()[0] + 1.) / 2.)
            processed_input_adv.save(filename,
                                     "JPEG",
                                     quality=FLAGS.JPEGquality)
            processed_input_adv = Image.open(filename)
            processed_input_adv = T2(processed_input_adv)
            processed_input_adv = (processed_input_adv.unsqueeze(0) * 2.) - 1.
            mixup_img_adv = processed_input_adv.to(device)

        pred_cle_mixup = classifier(mixup_img_cle)
        pred_cle_mixup_all = pred_cle_mixup_all + soft_max(pred_cle_mixup.data)

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
