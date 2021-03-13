import os
from pathlib import Path

IMAGES_IN_FILE = "images.txt"
TRAIN_SPLIT_IN_FILE = "train_test_split.txt"
IMAGES_OUT_FILE = "images.txt"
TRAIN_SPLIT_OUT_FILE = "train_test_split.txt"
REMOVED_LIST_OUT_FILE = "removed.txt"
IMAGES_FOLDER = "images"


# Look for image files that might have been removed and remove them from the images and train list files
# Useful to manually go through all images after downloading and manually remove those that aren't wanted
def remove_images_not_found(path_to_images_in_file, path_to_train_split_in_file, path_to_images_out_file, path_to_train_split_out_file, path_to_removed_file, images_folder):
    image_train = {}
    image_locations = {}
    with open(path_to_images_in_file, 'r') as in_images_file:
        with open(path_to_train_split_in_file, 'r') as in_train_split_file:
            images_lines = in_images_file.readlines()
            train_lines = in_train_split_file.readlines()

            for line in images_lines:
                uuid, image_location = line.strip('\n').split(' ', 2)
                image_locations[uuid] = image_location
            for line in train_lines:
                uuid, train_toggle = line.strip('\n').split(' ', 2)
                image_train[uuid] = int(train_toggle)
    with open(path_to_images_out_file, 'w') as out_images_file:
        with open(path_to_train_split_out_file, 'w') as out_train_split_file:
            with open(path_to_removed_file, 'w') as out_removed_file:
                for image_idx in image_locations:
                    if Path(os.path.join(images_folder, image_locations[image_idx])).is_file():
                        out_images_file.write(image_idx + ' ' + image_locations[image_idx] + '\n')
                        out_train_split_file.write(image_idx + ' ' + str(image_train[image_idx]) + '\n')
                    else:
                        out_removed_file.write(image_idx + ' ' + image_locations[image_idx] + ' ' + str(image_train[image_idx]) + '\n')

if __name__ == '__main__':
    remove_images_not_found(IMAGES_IN_FILE, TRAIN_SPLIT_IN_FILE, IMAGES_OUT_FILE, TRAIN_SPLIT_OUT_FILE, REMOVED_LIST_OUT_FILE, IMAGES_FOLDER)