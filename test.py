import numpy as np
import torch
import torch.utils.data as td
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
from torch.autograd import Variable
import utils
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import os


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = "results/yourmodel.pt"
CLASSES_FILE = "classes.txt"
NUM_CLASSES = 34
IN_DIR_DATA = "data"
IMAGE_ROWS = 4
IMAGE_COLUMNS = 4
SIZE_IMAGE = 375


def predict_image(image, transforms, model, device):
    image_tensor = transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(output)
    percentage = float(probabilities[0, index].item())
    return index, percentage


def get_random_images(num, data_dir, transforms):
    data = utils.DatasetBirds(data_dir, transform=transforms, train=False)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = td.DataLoader(dataset=data, sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels


def main_func(classes_file, in_dir_data, image_rows, image_columns, device, num_classes, model_in, size_image):
    # fill padded area with ImageNet's mean pixel value converted to range [0, 255]
    max_padding = tv.transforms.Lambda(utils.pad_function)

    # transform images
    transforms_test = tv.transforms.Compose([
        max_padding,
        tv.transforms.CenterCrop((size_image, size_image)),
        tv.transforms.ToTensor()
    ])

    # instantiate the model
    model = tv.models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # use the chosen model snapshot
    model.load_state_dict(torch.load(model_in, map_location=device))
    model.eval()

    # Get class names
    classnames = []
    path_to_classes = os.path.join(in_dir_data, classes_file)
    with open(path_to_classes, 'r') as classes_in_file:
        for class_line in classes_in_file:
            index, classname = class_line.strip('\n').split(' ', 1)
            classnames.append(classname.replace("_"," ").title())

    # generate image with the test set
    to_pil = tv.transforms.ToPILImage()
    images, labels = get_random_images(image_rows*image_columns, in_dir_data, transforms_test)
    fig = plt.figure(figsize=(20, 20))
    for ii in range(image_rows):
        for jj in range(image_columns):
            image = to_pil(images[ii + (jj * image_rows)])
            index, percentage = predict_image(image, transforms_test, model, device)
            sub = fig.add_subplot(image_rows, image_columns, ii + (jj * image_rows) + 1)
            sub.set_title("Prediction: " + classnames[index] + " (" + "{:.2f}".format(percentage*100) + "%)\nTruth: " + classnames[int(labels[ii + (jj * image_rows)])])
            plt.axis('off')
            plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main_func(CLASSES_FILE, IN_DIR_DATA, IMAGE_ROWS, IMAGE_COLUMNS, DEVICE, NUM_CLASSES, MODEL, SIZE_IMAGE)