import random
import uuid
import numpy as np
import os
import time
from pathlib import Path
import requests
from PIL import Image
from flickrapi import FlickrAPI


NUMBER_OF_IMAGES = 100
SEARCH_TERMS_IN_FILE = "search_queries.txt"
CLASSES_OUT_FILE = "classes.txt"
IMAGES_OUT_FILE = "images.txt"
TRAIN_SPLIT_OUT_FILE = "train_test_split.txt"
CLASS_SHIFT = 0
FLICKR_KEY = 'e5e1d15817b57d94d36214888c40095c'
FLICKR_SECRET = '740b8d260a5699a0'
IMAGES_FOLDER = "images"


# Download a specific Flickr photo and place it in the right class folder
def download_uri(uri, dir='./'):
    # Download
    f = dir + os.path.basename(uri)  # filename
    with open(f, 'wb') as file:
        file.write(requests.get(uri, timeout=10).content)

    # Rename (remove wildcard characters)
    src = f  # original name
    for c in ['%20', '%', '*', '~', '(', ')']:
        f = f.replace(c, '_')
    f = f[:f.index('?')] if '?' in f else f  # new name
    if src != f:
        os.rename(src, f)  # rename

    # Add suffix (if missing)
    if Path(f).suffix == '':
        src = f  # original name
        f += '.' + Image.open(f).format.lower()  # append PIL format
        os.rename(src, f)  # rename


# Get the download links through the Flickr API and store them in the images folder by class.
def get_urls(key, secret, images_folder, search='honeybees on flowers', n=10, download=False):
    t = time.time()
    flickr = FlickrAPI(key, secret)
    license = ()  # https://www.flickr.com/services/api/explore/?method=flickr.photos.licenses.getInfo
    photos = flickr.walk(text=search,  # http://www.flickr.com/services/api/flickr.photos.search.html
                         extras='url_o',
                         per_page=50,  # 1-500
                         license=license,
                         sort='relevance')

    if download:
        dir = os.getcwd() + os.sep + images_folder + os.sep + search.replace(' ', '_') + os.sep  # save directory
        if not os.path.exists(dir):
            os.makedirs(dir)

    urls = []
    for i, photo in enumerate(photos):
        if i < n:
            try:
                # construct url https://www.flickr.com/services/api/misc.urls.html
                url = photo.get('url_o')  # original size
                if url is None:
                    url = 'https://farm%s.staticflickr.com/%s/%s_%s_b.jpg' % \
                          (photo.get('farm'), photo.get('server'), photo.get('id'), photo.get('secret'))  # large size

                # download
                if download:
                    download_uri(url, dir)

                urls.append(url)
                print('%g/%g %s' % (i+1, n, url))
            except:
                print('%g/%g error...' % (i, n))
        else:
            break


# Download a set of images per class defined by the search terms list. As well as create the classes text file.
def download_images(search_terms_file, classes_file, class_shift, number_of_images, flickr_key, flickr_secret, images_folder):
    with open(search_terms_file, 'r') as search_terms_in_file:
        with open(classes_file, 'w') as classes_out_file:
            for index, search_term in enumerate(search_terms_in_file, 1):
                get_urls( key=flickr_key, secret=flickr_secret, images_folder=images_folder, search=search_term.strip('\n'), n=number_of_images, download=True)
                classes_out_file.write(str(index + class_shift).zfill(4) + " " + search_term.replace(" ", "_"))
                os.rename(os.path.join(images_folder, search_term.strip('\n').replace(" ", "_")), os.path.join(images_folder, str(index + class_shift).zfill(4)))


# Create image list with unique IDs and filename as well as a random split of images used for training or testing
def make_images_and_train_split_files(images_file, train_split_file, images_folder):
    with open(images_file, 'w') as images_out_file:
        with open(train_split_file, 'w') as train_split_out_file:
            for path, subdirs, files in os.walk(images_folder):
                for subdir in subdirs:
                    _, _, subdir_files = next(os.walk(os.path.join(images_folder, subdir)))
                    random_list = [1] * (int(np.floor(len(subdir_files)/2))) + [0] * (int(len(subdir_files) - np.floor(len(subdir_files)/2)))
                    random.shuffle(random_list)
                    for index, file in enumerate(subdir_files):
                        new_uuid = str(uuid.uuid4())
                        images_out_file.write(new_uuid + ' ' + os.path.join(os.path.join(images_folder, subdir), file).split("\\", 1)[1].replace("\\", "/") + '\n')
                        train_split_out_file.write(new_uuid + ' ' + str(random_list[index]) + '\n')


if __name__ == '__main__':
    download_images(SEARCH_TERMS_IN_FILE, CLASSES_OUT_FILE, CLASS_SHIFT, NUMBER_OF_IMAGES, FLICKR_KEY, FLICKR_SECRET, IMAGES_FOLDER)
    make_images_and_train_split_files(IMAGES_OUT_FILE, TRAIN_SPLIT_OUT_FILE, IMAGES_FOLDER)