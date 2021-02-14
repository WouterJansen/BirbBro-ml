import os
import shutil
import uuid


IN_DIR_DATA = "../data_cub_200_2011"
IMAGES_FOLDER = "images"
IMAGES_FILE = 'images.txt'
SPLITS_FILE = "train_test_split.txt"
CLASSES_FILE = "classes.txt"
IMAGES_REMOVAL_FILE = 'images_removed.txt'
SPLITS_REMOVAL_FILE = "train_test_split_removed.txt"
CLASSES_REMOVAL_FILE = "classes_removed.txt"
IMAGES_CONVERTED_FILE = 'images_converted.txt'
SPLITS_CONVERTED_FILE = "train_test_split_converted.txt"
IMAGES_RENAMED_FILE = 'images_renamed.txt'
CLASSES_RENAMED_FILE = "classes_renamed.txt"
UNWANTED_SPECIES_FILE = "remove_list_cub_200_2011.txt"
CLASS_SHIFT = 0


# Remove classes that aren't wanted to lower trained features count
def remove_unwanted_species(splits_file, images_file, classes_file, splits_removal_file, images_removal_file,
                            classes_removal_file, unwanted_species_file, in_dir_data, images_folder):
    path_to_splits = os.path.join(in_dir_data, splits_file)
    path_to_images = os.path.join(in_dir_data, images_file)
    path_to_classes = os.path.join(in_dir_data, classes_file)
    path_to_new_splits = os.path.join(in_dir_data, splits_removal_file)
    path_to_new_images = os.path.join(in_dir_data, images_removal_file)
    path_to_new_classes = os.path.join(in_dir_data, classes_removal_file)
    path_to_removals = os.path.join('.', unwanted_species_file)
    img_root = os.path.join(in_dir_data, images_folder)

    with open(path_to_removals, 'r') as in_removal_file:
        image_files = []
        class_folders = []
        with open(path_to_classes, 'r') as in_classes_file:
            classes_lines = in_classes_file.readlines()
        for line in in_removal_file:
            class_folder = os.path.join(img_root, line.strip('\n'))
            class_folders.append(line.strip('\n'))
            image_files.extend([f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))])
            shutil.rmtree(class_folder)
        with open(path_to_new_classes, 'w') as out_classes_file:
            for class_line in classes_lines:
                class_idx, classname = class_line.strip('\n').split(' ', 2)
                if not any(class_folder in classname for class_folder in class_folders):
                    out_classes_file.write(class_idx + ' ' + classname + '\n')
        with open(path_to_images, 'r') as in_images_file:
            image_lines = in_images_file.readlines()
        with open(path_to_splits, 'r') as in_splits_file:
            split_lines = in_splits_file.readlines()
        with open(path_to_new_images, 'w') as out_images_file:
            with open(path_to_new_splits, 'w') as out_splits_file:
                for index, line in enumerate(image_lines, 1):
                    image_idx, filename = line.strip('\n').split(' ', 2)
                    split_idx, use_train = split_lines[index - 1].strip('\n').split(' ', 2)
                    if image_idx != split_idx:
                        print("index of image and split files don't match, this is bad!")
                    if not any(image_file in filename for image_file in image_files):
                        out_splits_file.write(image_idx + ' ' + use_train + '\n')
                        out_images_file.write(image_idx + ' ' + filename + '\n')


# Convert image ID numbers to UUIDs similar to NaBirds
def convert_ids(splits_file, images_file, splits_converted_file, images_converted_file, in_dir_data):
    path_to_splits = os.path.join(in_dir_data, splits_file)
    path_to_images = os.path.join(in_dir_data, images_file)
    path_to_new_splits = os.path.join(in_dir_data, splits_converted_file)
    path_to_new_images = os.path.join(in_dir_data, images_converted_file)

    with open(path_to_images, 'r') as in_images_file:
        with open(path_to_splits, 'r') as in_splits_file:
            with open(path_to_new_images, 'w') as out_images_file:
                with open(path_to_new_splits, 'w') as out_splits_file:
                    split_lines = in_splits_file.readlines()
                    for index, line in enumerate(in_images_file, 1):
                        new_uuid = str(uuid.uuid4())
                        split_idx, use_train = split_lines[index-1].strip('\n').split(' ', 2)
                        image_idx, filename = line.strip('\n').split(' ', 2)
                        if image_idx != split_idx:
                            print("index of image and split files don't match, this is bad!")
                        out_splits_file.write(new_uuid + ' ' + use_train + '\n')
                        out_images_file.write(new_uuid + ' ' + filename + '\n')


# Rename class folders to simple class number like NaBirds (with 4 digits)
def rename_class_folders_to_indexes(classes_file, classes_renamed_file, images_file, images_renamed_file,
                                    in_dir_data, images_folder, class_shift):
    path_to_classes = os.path.join(in_dir_data, classes_file)
    path_to_new_classes = os.path.join(in_dir_data, classes_renamed_file)
    path_to_images = os.path.join(in_dir_data, images_file)
    path_to_new_images = os.path.join(in_dir_data, images_renamed_file)
    img_root = os.path.join(in_dir_data, images_folder)

    with open(path_to_classes, 'r') as in_classes_file:
        classes_lines = in_classes_file.readlines()
    new_indexes = {}
    with open(path_to_new_classes, 'w') as out_classes_file:
        for index, class_line in enumerate(classes_lines, 1):
            new_index = str(index + class_shift).zfill(4)
            class_idx, classname = class_line.strip('\n').split(' ', 2)
            new_indexes[int(class_idx)] = new_index
            out_classes_file.write(new_index + ' ' + classname.split(".")[1] + '\n')
            if os.path.exists(os.path.join(img_root, classname)):
                os.rename(os.path.join(img_root, classname), os.path.join(img_root, new_index))
    with open(path_to_images, 'r') as in_images_file:
        with open(path_to_new_images, 'w') as out_images_file:
            for index, image_line in enumerate(in_images_file, 1):
                image_idx, filename = image_line.strip('\n').split(' ', 2)
                class_folder, imagename = filename.split("/", 1)
                class_index = class_folder[0:3]
                out_images_file.write(image_idx + ' ' + new_indexes[int(class_index)]  + "/" + imagename + '\n')


# Verify that all images listed are available in the listed class folder
def verify_images(image_file, in_dir_data, images_folder):
    path_to_images = os.path.join(in_dir_data, image_file)
    img_root = os.path.join(in_dir_data, images_folder)
    with open(path_to_images, 'r') as in_images_file:
        for index, image_line in enumerate(in_images_file, 1):
            given_uuid, filename = image_line.strip('\n').split(' ', 1)
            if not os.path.isfile(os.path.join(img_root, filename)):
                print(filename + " is missing!")


if __name__ == '__main__':
    remove_unwanted_species(SPLITS_FILE, IMAGES_FILE, CLASSES_FILE, SPLITS_REMOVAL_FILE, IMAGES_REMOVAL_FILE,
                            CLASSES_REMOVAL_FILE, UNWANTED_SPECIES_FILE, IN_DIR_DATA, IMAGES_FOLDER)
    convert_ids(SPLITS_REMOVAL_FILE, IMAGES_REMOVAL_FILE, SPLITS_CONVERTED_FILE, IMAGES_CONVERTED_FILE, IN_DIR_DATA)
    rename_class_folders_to_indexes(CLASSES_REMOVAL_FILE, CLASSES_RENAMED_FILE, IMAGES_CONVERTED_FILE,
                                    IMAGES_RENAMED_FILE, IN_DIR_DATA, IMAGES_FOLDER, CLASS_SHIFT)
    verify_images(IMAGES_RENAMED_FILE, IN_DIR_DATA, IMAGES_FOLDER)