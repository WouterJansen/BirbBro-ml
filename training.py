import os
import numpy as np
import sklearn.model_selection as skms
import sklearn.metrics as skm
import torch
import torch.utils.data as td
import torch.nn.functional as F
import torch.nn as nn
import torchvision as tv
import utils
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR_RESULTS = 'results'
RANDOM_SEED = 42
IN_DIR_DATA = "data"
BATCH_SIZE = 24
WORKERS = 6
NUM_EPOCHS = 40
NUM_CLASSES = 34
SIZE_IMAGE = 375


def main_func(device, out_dir_results, random_seed, in_dir_data, batch_size, workers, num_epochs, num_classes, size_image):
    # create an output folder
    os.makedirs(OUT_DIR_RESULTS, exist_ok=True)

    # fill padded area with ImageNet's mean pixel value converted to range [0, 255]
    max_padding = tv.transforms.Lambda(utils.pad_function)

    # transform images
    transforms_train = tv.transforms.Compose([
        max_padding,
        tv.transforms.RandomOrder([
            tv.transforms.RandomCrop((size_image, size_image)),
            tv.transforms.RandomHorizontalFlip()
        ]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms_eval = tv.transforms.Compose([
        max_padding,
        tv.transforms.CenterCrop((size_image, size_image)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # instantiate dataset objects according to the pre-defined splits
    ds_train = utils.DatasetBirds(in_dir_data, transform=transforms_train, train=True)
    ds_val = utils.DatasetBirds(in_dir_data, transform=transforms_eval, train=True)
    ds_test = utils.DatasetBirds(in_dir_data, transform=transforms_eval, train=False)

    splits = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_seed)
    idx_train, idx_val = next(splits.split(np.zeros(len(ds_train)), ds_train.targets))

    # set hyper-parameters
    params = {'batch_size': batch_size, 'num_workers': workers}

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

    model_desc = utils.get_model_desc(num_classes=num_classes, pretrained=True)

    # instantiate the model
    model = tv.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

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

            x = x.to(device)
            y = y.to(device)

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

                x = x.to(device)
                y = y.to(device)

                # predict bird species
                y_pred = model(x)

                # calculate the loss
                loss = F.cross_entropy(y_pred, y)

                # calculate the accuracy
                acc = skm.accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])

                val_loss.append(loss.item())
                val_acc.append(acc)

            val_acc_avg.append(np.mean(val_acc))

            # save the best model snapshot
            current_val_acc = val_acc_avg[-1]
            if current_val_acc > best_val_acc:
                if best_snapshot_path is not None:
                    os.remove(best_snapshot_path)

                best_val_acc = current_val_acc
                best_snapshot_path = os.path.join(out_dir_results, f'model_{model_desc}_ep={epoch}_acc={best_val_acc}.pt')

                torch.save(model.state_dict(), best_snapshot_path)

        # adjust the learning rate
        scheduler.step()

        # print performance metrics
        print('Epoch {} |> Train. loss: {:.4f} | Val. loss: {:.4f}'.format(epoch + 1, np.mean(train_loss), np.mean(val_loss)))

    # use the best model snapshot
    model.load_state_dict(torch.load(best_snapshot_path, map_location=device))

    # test the model
    true = list()
    pred = list()
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            true.extend([val.item() for val in y])
            pred.extend([val.item() for val in y_pred.argmax(dim=-1)])

    # calculate the accuracy
    test_accuracy = skm.accuracy_score(true, pred)

    # save the accuracy
    path_to_logs = f'{out_dir_results}/logs.csv'
    utils.log_accuracy(path_to_logs, model_desc, test_accuracy)

    print('Test accuracy: {:.3f}'.format(test_accuracy))


if __name__ == '__main__':
    main_func(DEVICE, OUT_DIR_RESULTS, RANDOM_SEED, IN_DIR_DATA, BATCH_SIZE, WORKERS, NUM_EPOCHS, NUM_CLASSES, SIZE_IMAGE)