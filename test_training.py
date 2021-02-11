# import packages
import os
import csv
import random
import tarfile
import multiprocessing as mp

import tqdm
import requests

import numpy as np
import sklearn.model_selection as skms

import torch
import torch.utils.data as td
import torch.nn.functional as F
import torch.nn as nn

import torchvision as tv
import torchvision.transforms.functional as TF

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# define constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = 'results'
RANDOM_SEED = 42

# create an output folder
os.makedirs(OUT_DIR, exist_ok=True)


def get_model_desc(pretrained=False, num_classes=200, use_attention=False):
    """
    Generates description string.
    """
    desc = list()

    if pretrained:
        desc.append('Transfer')
    else:
        desc.append('Baseline')

    if num_classes == 204:
        desc.append('Multitask')

    if use_attention:
        desc.append('Attention')

    return '-'.join(desc)


def log_accuracy(path_to_csv, desc, acc, sep='\t', newline='\n'):
    """
    Logs accuracy into a CSV-file.
    """
    file_exists = os.path.exists(path_to_csv)

    mode = 'a'
    if not file_exists:
        mode += '+'

    with open(path_to_csv, mode) as csv:
        if not file_exists:
            csv.write(f'setup{sep}accuracy{newline}')

        csv.write(f'{desc}{sep}{acc}{newline}')


def get_filepaths(path_to_data, fileformat='.jpg'):
    """
    Ruturns paths to files of the specified format.
    """
    filepaths = list()
    for root, _, finenames in os.walk(path_to_data):
        for fn in finenames:
            if fn.endswith(fileformat):
                filepaths.append(os.path.join(root, fn))

    return filepaths


def cleaning_worker(path_to_img):
    """
    Verifies whether the image is corrupted.
    """
    std = np.std(mpimg.imread(path_to_img))
    img_ok = not np.isclose(std, 0.0)

    return img_ok, path_to_img


class DatasetBirds(tv.datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset.
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 loader=tv.datasets.folder.default_loader,
                 train=True,
                 bboxes=False):

        img_root = os.path.join(root, 'images')

        super(DatasetBirds, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader
        )

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train

        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, 'train_test_split.txt')
        indices_to_use = list()
        with open(path_to_splits, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images
        path_to_index = os.path.join(root, 'images.txt')
        filenames_to_use = set()
        with open(path_to_index, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(os.path.join(img_root, fn.replace("/","\\")))

        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, 'bounding_boxes.txt')
            bounding_boxes = list()
            with open(path_to_bboxes, 'r') as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(lambda x: float(x), line.strip('\n').split(' '))
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(DatasetBirds, self).__getitem__(index)

        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target


def pad(img, fill=0, size_max=500):
    """
    Pads images to the specified size (height x width).
    Fills up the padded area with value(s) passed to the `fill` parameter.
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


if __name__ == '__main__':

    in_dir_data = "data"

    # fill padded area with ImageNet's mean pixel value converted to range [0, 255]
    fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
    max_padding = tv.transforms.Lambda(lambda x: pad(x, fill=fill))

    # transform images
    transforms_train = tv.transforms.Compose([
        max_padding,
        tv.transforms.RandomOrder([
            tv.transforms.RandomCrop((375, 375)),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomVerticalFlip()
        ]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_eval = tv.transforms.Compose([
        max_padding,
        tv.transforms.CenterCrop((375, 375)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # instantiate dataset objects according to the pre-defined splits
    ds_train = DatasetBirds(in_dir_data, transform=transforms_train, train=True)
    ds_val = DatasetBirds(in_dir_data, transform=transforms_eval, train=True)
    ds_test = DatasetBirds(in_dir_data, transform=transforms_eval, train=False)

    splits = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED)
    idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))

    # set hyper-parameters
    params = {'batch_size': 24, 'num_workers': 8}
    pretrained = True
    num_epochs = 100
    num_classes = 200

    # instantiate data loaders
    train_loader = td.DataLoader(
        dataset=ds_train,
        sampler=td.SubsetRandomSampler(idx_train),
        **params
    )
    val_loader = td.DataLoader(
        dataset=ds_val,
        sampler=td.SubsetRandomSampler(idx_val),
        **params
    )
    test_loader = td.DataLoader(dataset=ds_test, **params)

    model_desc = get_model_desc(num_classes=num_classes, pretrained=pretrained)

    # instantiate the model
    model = tv.models.resnet50(pretrained=True).to(DEVICE)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # instantiate optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # train and validate the model
    best_snapshot_path = None
    val_acc_avg = list()
    best_val_acc = -1.0

    for epoch in range(num_epochs):

        # train the model
        model.train()
        train_loss = list()
        for batch in train_loader:
            x, y = batch

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            # calculate the loss
            y_pred = model(x)

            # calculate the loss
            loss = F.cross_entropy(y_pred, y)

            # backprop & update weights
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        # validate the model
        model.eval()
        val_loss = list()
        val_acc = list()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                # predict bird species
                y_pred = model(x)

                # calculate the loss
                loss = F.cross_entropy(y_pred, y)

                # calculate the accuracy
                acc = skms.accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])

                val_loss.append(loss.item())
                val_acc.append(acc)

            val_acc_avg.append(np.mean(val_acc))

            # save the best model snapshot
            current_val_acc = val_acc_avg[-1]
            if current_val_acc > best_val_acc:
                if best_snapshot_path is not None:
                    os.remove(best_snapshot_path)

                best_val_acc = current_val_acc
                best_snapshot_path = os.path.join(OUT_DIR, f'model_{model_desc}_ep={epoch}_acc={best_val_acc}.pt')

                torch.save(model.state_dict(), best_snapshot_path)

        # adjust the learning rate
        scheduler.step()

        # print performance metrics
        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            print('Epoch {} |> Train. loss: {:.4f} | Val. loss: {:.4f}'.format(
                epoch + 1, np.mean(train_loss), np.mean(val_loss))
            )

    # use the best model snapshot
    model.load_state_dict(torch.load(best_snapshot_path, map_location=DEVICE))

    # test the model
    true = list()
    pred = list()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_pred = model(x)

            true.extend([val.item() for val in y])
            pred.extend([val.item() for val in y_pred.argmax(dim=-1)])

    # calculate the accuracy
    test_accuracy = skms.accuracy_score(true, pred)

    # save the accuracy
    path_to_logs = f'{OUT_DIR}/logs.csv'
    log_accuracy(path_to_logs, model_desc, test_accuracy)

    print('Test accuracy: {:.3f}'.format(test_accuracy))