import sys
import os
import gzip
import numpy as np
import pickle
import scipy.io as sio
import h5py


def to_one_hot(x, depth):
    ret = np.zeros((x.shape[0], depth), dtype=np.int32)
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def get_ssl_data(x_train, y_train, num_labeled, n_y, seed=None):
    if num_labeled is None:
        return x_train, y_train

    seed = int(seed)
    rng_data = np.random.RandomState(seed)
    inds = rng_data.permutation(x_train.shape[0])
    x_train = x_train[inds]
    y_train = y_train[inds]

    x_labelled = []
    y_labelled = []
    for j in range(n_y):
        x_labelled.append(x_train[y_train == j][:num_labeled // n_y])
        y_labelled.append(y_train[y_train == j][:num_labeled // n_y])
    x_train = np.concatenate(x_labelled)
    y_train = np.concatenate(y_labelled)
    return x_train, y_train


def load_mnist_realval(path="/home/Data/mnist.pkl.gz",
                       asimage=True,
                       one_hot=False,
                       validation=True,
                       isTf=True,
                       return_all=False,
                       **kwargs):
    """
    return_all flag will return all of the data. It will overwrite validation
    nlabeled.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)

        def download_dataset(url, _path):
            print("Downloading data from %s" % url)
            if sys.version_info > (2, ):
                import urllib.request as request
            else:
                from urllib2 import Request as request
            request.urlretrieve(url, _path)

        download_dataset(
            "http://www.iro.umontreal.ca/~lisa/deep/data/mnist"
            "/mnist.pkl.gz", path)

    with gzip.open(path, "rb") as f:
        if sys.version_info > (3, ):
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        else:
            train_set, valid_set, test_set = pickle.load(f)
    x_train, y_train = train_set[0], train_set[1].astype("int32")
    x_valid, y_valid = valid_set[0], valid_set[1].astype("int32")
    x_test, y_test = test_set[0], test_set[1].astype("int32")

    n_y = y_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    y_train, y_valid = t_transform(y_train), t_transform(y_valid)
    y_test = t_transform(y_test)

    if asimage is True:
        x_train = x_train.reshape([-1, 28, 28, 1])
        x_valid = x_valid.reshape([-1, 28, 28, 1])
        x_test = x_test.reshape([-1, 28, 28, 1])
    if isTf is False:
        x_train = x_train.transpose([0, 3, 1, 2])
        x_valid = x_valid.transpose([0, 3, 1, 2])
        x_test = x_test.transpose([0, 3, 1, 2])

    if return_all is True:
        return x_train, y_train, x_valid, y_valid, x_test, y_test

    if validation is True:
        x_test = x_valid
        y_test = y_valid
    else:
        x_train = np.concatenate((x_train, x_valid))
        y_train = np.concatenate((y_train, y_valid))

    return x_train, y_train, x_test, y_test


def load_cifar10(data_dir="/home/Data/cifar/",
                 one_hot=False,
                 isTf=True,
                 **kwargs):
    def file_name(ind):
        return os.path.join(data_dir,
                            "cifar-10-batches-py/data_batch_" + str(ind))

    def unpickle_cifar_batch(file_):
        fo = open(file_, "rb")
        tmp_data = pickle.load(fo, encoding="bytes")
        fo.close()
        x_ = tmp_data[b"data"].astype(np.float32)
        x_ = x_.reshape((10000, 3, 32, 32)) / 255.0
        y_ = np.array(tmp_data[b"labels"]).astype(np.float32)
        return {"x": x_, "y": y_}

    train_data = [unpickle_cifar_batch(file_name(i)) for i in range(1, 6)]
    x_train = np.concatenate([td["x"] for td in train_data])
    y_train = np.concatenate([td["y"] for td in train_data])
    y_train = y_train.astype("int32")

    test_data = unpickle_cifar_batch(
        os.path.join(data_dir, "cifar-10-batches-py/test_batch"))
    x_test = test_data["x"]
    y_test = test_data["y"].astype("int32")

    n_y = int(y_test.max() + 1)
    y_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)

    if isTf is True:
        x_train = x_train.transpose([0, 2, 3, 1])
        x_test = x_test.transpose([0, 2, 3, 1])
    return x_train, y_transform(y_train), x_test, y_transform(y_test)


def load_cifar100(data_dir="/home/Data/cifar/",
                  one_hot=False,
                  isTf=True,
                  **kwargs):
    def unpickle_cifar_batch(file_, num):
        fo = open(file_, "rb")
        if sys.version_info > (3, ):
            tmp_data = pickle.load(fo, encoding="bytes")
        else:
            tmp_data = pickle.load(fo)
        fo.close()
        x_ = tmp_data[b"data"].astype(np.float32)
        x_ = x_.reshape((num, 3, 32, 32)) / 255.0
        y_ = np.array(tmp_data[b"fine_labels"]).astype(np.float32)
        return {"x": x_, "y": y_}

    train_data = unpickle_cifar_batch(os.path.join(data_dir,
                                                   "cifar-100-python/train"),
                                      num=50000)
    x_train = train_data["x"]
    y_train = train_data["y"].astype("int32")

    test_data = unpickle_cifar_batch(os.path.join(data_dir,
                                                  "cifar-100-python/test"),
                                     num=10000)
    x_test = test_data["x"]
    y_test = test_data["y"].astype("int32")

    n_y = int(y_test.max() + 1)
    y_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)

    if isTf is True:
        x_train = x_train.transpose([0, 2, 3, 1])
        x_test = x_test.transpose([0, 2, 3, 1])
    return x_train, y_transform(y_train), x_test, y_transform(y_test)


def load_svhn(data_dir="/home/Data", one_hot=False, isTf=True, **kwargs):
    data_dir = os.path.join(data_dir, "svhn")
    train_dat = sio.loadmat(os.path.join(data_dir, "train_32x32.mat"))
    train_x = train_dat["X"].astype("float32")
    train_y = train_dat["y"].flatten()
    train_y[train_y == 10] = 0
    train_x = train_x.transpose([3, 0, 1, 2])

    test_dat = sio.loadmat(os.path.join(data_dir, "test_32x32.mat"))
    test_x = test_dat["X"].astype("float32")
    test_y = test_dat["y"].flatten()
    test_y[test_y == 10] = 0
    test_x = test_x.transpose([3, 0, 1, 2])

    n_y = int(train_y.max() + 1)
    y_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)

    if isTf is False:
        train_x = train_x.transpose([0, 3, 1, 2])
        test_x = test_x.transpose([0, 3, 1, 2])

    train_x, test_x = train_x / 255.0, test_x / 255.0
    return train_x, y_transform(train_y), test_x, y_transform(test_y)


def load_caltech101_img(path="./Data/",
                        require_subset=['img', 'seg'],
                        num_train_test=[30, 20],
                        seed=1234):
    with open(os.path.join(path, "Caltech101_label2index.pkl"), "rb") as f:
        label_index_dict = pickle.load(f)

    results_list = [[] for _ in require_subset]
    label_list = []
    hf = h5py.File(os.path.join(path, "Caltech101.h5"), 'r')
    for k in hf.keys():
        if k not in label_index_dict:
            continue
        label = label_index_dict[k]
        ind = 0
        while str(ind) in hf[k].keys():
            for require_ind, require_key in enumerate(require_subset):
                results_list[require_ind].append(hf[k][str(ind) + "_" +
                                                       require_key])
            label_list.append(label)
            ind = ind + 1

    # label_list = np.asarray(label_list)
    # results_list = [np.asarray(item) for item in results_list]

    rng_data = np.random.RandomState(seed)
    inds = rng_data.permutation(len(label_list))
    new_results_list = [[] for _ in require_subset]
    new_label_list = []
    for ind in inds:
        for data_id, data_item in enumerate(results_list):
            new_results_list[data_id].append(data_item[ind])
        new_label_list.append(label_list[ind])

    results_list = new_results_list
    label_list = np.array(new_label_list)

    results_train, label_train = [[] for _ in require_subset], []
    results_test, label_test = [[] for _ in require_subset], []
    total_num = num_train_test[0] + num_train_test[1]

    for label_ind in range(np.max(label_list) + 1):
        index_each = np.where(label_list == label_ind)[0]
        for count, ind_each in enumerate(index_each):
            if count < num_train_test[1]:
                for data_id, data_item in enumerate(results_list):
                    results_test[data_id].append(data_item[ind_each])
                label_test.append(label_ind)
            elif count < total_num:
                for data_id, data_item in enumerate(results_list):
                    results_train[data_id].append(data_item[ind_each])
                label_train.append(label_ind)
            else:
                break

    return results_train, label_train, results_test, label_test


def load_caltech101_path(path="./Data/", num_train_test=[30, 20], seed=1234):
    with open(os.path.join(path, "Caltech101_imgpath.pkl"), "rb") as f:
        path_label_dict = pickle.load(f)

    imgpath_list = path_label_dict['img_path']
    label_list = path_label_dict['label']

    # label_list = np.asarray(label_list)
    # results_list = [np.asarray(item) for item in results_list]

    rng_data = np.random.RandomState(seed)
    inds = rng_data.permutation(len(label_list))
    new_imgpath_list = []
    new_label_list = []
    for ind in inds:
        new_imgpath_list.append(imgpath_list[ind])
        new_label_list.append(label_list[ind])

    imgpath_list = new_imgpath_list
    label_list = np.array(new_label_list)

    results_train, label_train = [], []
    results_test, label_test = [], []
    total_num = num_train_test[0] + num_train_test[1]

    for label_ind in range(np.max(label_list) + 1):
        index_each = np.where(label_list == label_ind)[0]
        for count, ind_each in enumerate(index_each):
            if count < num_train_test[1]:
                results_test.append(imgpath_list[ind_each])
                label_test.append(label_ind)
            elif count < total_num:
                results_train.append(imgpath_list[ind_each])
                label_train.append(label_ind)
            else:
                break

    return results_train, label_train, results_test, label_test


def load_imagenet_path(path="./Data/", number_of_classes=1000):
    with open(os.path.join(path, "ImageNet_imgpath.pkl"), "rb") as f:
        path_label_dict = pickle.load(f)

    train_img_path_label = path_label_dict['train']
    valid_img_path_label = path_label_dict['validation']

    def get_subset(img_path_label):
        imgpath_list = img_path_label['img_path']
        label_list = img_path_label['label']
        label_list = np.array(label_list)

        img_list, lab_list = [], []

        for label_ind in range(number_of_classes):
            index_each = np.where(label_list == label_ind)[0]
            for count, ind_each in enumerate(index_each):
                img_list.append(imgpath_list[ind_each])
                lab_list.append(label_ind)

        return img_list, lab_list

    train_img, train_label = get_subset(train_img_path_label)
    test_img, test_label = get_subset(valid_img_path_label)
    return train_img, train_label, test_img, test_label
