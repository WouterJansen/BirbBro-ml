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


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = "results/model_Transfer_ep=68_acc=0.825.pt"
NUM_CLASSES = 34
IN_DIR_DATA = "data"
IMAGE_ROWS = 5
IMAGE_COLUMNS = 5


def predict_image(image, transforms, model, device):
    image_tensor = transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index


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


def main_func():
    # fill padded area with ImageNet's mean pixel value converted to range [0, 255]
    max_padding = tv.transforms.Lambda(utils.pad_function)

    # transform images
    transforms_test = tv.transforms.Compose([
        max_padding,
        tv.transforms.CenterCrop((500, 500)),
        tv.transforms.ToTensor()
    ])

    # instantiate the model
    model = tv.models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(DEVICE)

    # use the chosen model snapshot
    model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
    model.eval()

    # generate image with the test set
    to_pil = tv.transforms.ToPILImage()
    images, labels = get_random_images(IMAGE_ROWS*IMAGE_COLUMNS, IN_DIR_DATA, transforms_test)
    fig = plt.figure(figsize=(15, 15))
    for ii in range(IMAGE_ROWS):
        for jj in range(IMAGE_COLUMNS):
            image = to_pil(images[ii + (jj * IMAGE_ROWS)])
            index = predict_image(image, transforms_test, model, DEVICE)
            sub = fig.add_subplot(IMAGE_ROWS, IMAGE_COLUMNS, ii + (jj * IMAGE_ROWS) + 1)
            sub.set_title("prediction:" + str(index) + " \ntruth:" + str(int(labels[ii + (jj * IMAGE_ROWS)])))
            plt.axis('off')
            plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main_func()