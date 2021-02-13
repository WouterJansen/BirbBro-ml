import os
import torchvision as tv
import torchvision.transforms.functional as TF


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


def convert_to_float(x):
    return float(x)


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
                 train=True):

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
                    if os.name == 'nt':
                        filenames_to_use.add(os.path.join(img_root, fn.replace("/", "\\")))
                    else:
                        filenames_to_use.add(fn)


        img_paths_cut = {'/'.join(img_path.rsplit('/', 2)[-2:]): idx for idx, (img_path, lb) in enumerate(self.imgs)}
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(DatasetBirds, self).__getitem__(index)

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

    return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=fill)


def pad_function(x):
    fill = tuple(map(tv.transforms.Lambda(int_round_and_multiply256), (0.485, 0.456, 0.406)))
    return pad(x, fill=fill)


def int_round_and_multiply256(x):
    return int(round(x * 256))