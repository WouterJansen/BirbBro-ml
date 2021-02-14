import os
import shutil
import uuid


IN_DIR_DATA = "../data_nabirds"
IMAGES_FOLDER = "images"
IMAGES_FILE = 'images.txt'
SPLITS_FILE = "train_test_split.txt"
CLASSES_FILE = "classes.txt"
IMAGES_REMOVAL_FILE = 'images_removed.txt'
SPLITS_REMOVAL_FILE = "train_test_split_removed.txt"
CLASSES_REMOVAL_FILE = "classes_removed.txt"
IMAGES_RENAMED_FILE = 'images_renamed.txt'
IMAGES_RENAMED_TEMP_FILE = 'images_renamed_temp.txt'
SPLITS_RENAMED_FILE = "train_test_split_renamed.txt"
CLASSES_RENAMED_FILE = "classes_renamed.txt"
CLASSES_RENAMED_TEMP_FILE = "classes_renamed_temp.txt"
UNWANTED_SPECIES_FILE = "remove_list_nabirds.txt"
IMAGES_MERGED_FILE = "images_merged.txt"
CLASSES_MERGED_FILE = "classes_merged.txt"
HIERARCHY_FILE = "hierarchy.txt"
HIERARCHY_MERGE_FILE = "merge_list_nabirds.txt"
CLASS_SHIFT = 92


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
    indexCounter = 1
    with open(path_to_new_classes, 'w') as out_classes_file:
        for index, class_line in enumerate(classes_lines, 1):
            class_idx, classname = class_line.strip('\n').split(' ', 2)
            if os.path.exists(os.path.join(img_root, classname)):
                new_index = str(indexCounter + class_shift).zfill(4)
                indexCounter += 1
                os.rename(os.path.join(img_root, classname), os.path.join(img_root, new_index))
                new_indexes[int(class_idx)] = new_index
                out_classes_file.write(new_index + ' ' + classname.split(".")[1] + '\n')
    with open(path_to_images, 'r') as in_images_file:
        with open(path_to_new_images, 'w') as out_images_file:
            for index, image_line in enumerate(in_images_file, 1):
                image_idx, filename = image_line.strip('\n').split(' ', 2)
                class_folder, imagename = filename.split("/", 1)
                class_index = class_folder[0:4]
                out_images_file.write(image_idx + ' ' + new_indexes[int(class_index)]  + "/" + imagename + '\n')


# Rename class folders to  class number (4 digits) and class string combination for easier readability
def rename_class_folders_to_class_names(classes_file, classes_renamed_file, images_file,
                                        images_renamed_file, in_dir_data, images_folder):
    path_to_classes = os.path.join(in_dir_data, classes_file)
    path_to_new_classes = os.path.join(in_dir_data, classes_renamed_file)
    path_to_images = os.path.join(in_dir_data, images_file)
    path_to_new_images = os.path.join(in_dir_data, images_renamed_file)
    img_root = os.path.join(in_dir_data, images_folder)

    with open(path_to_classes, 'r') as in_classes_file:
        classes_lines = in_classes_file.readlines()
    classes_dict = {}
    with open(path_to_new_classes, 'w') as out_classes_file:
        for index, class_line in enumerate(classes_lines, 1):
            class_idx, classname = class_line.strip('\n').split(' ', 1)
            out_classes_file.write(str(class_idx).zfill(4) + ' ' + str(class_idx).zfill(4)
                                   + "." + classname.replace(" ", "_").replace("/", "-") + "\n")
            classes_dict[str(class_idx).zfill(4)] = str(class_idx).zfill(4) \
                                                    + "." + classname.replace(" ", "_").replace("/", "-")
            if os.path.exists(os.path.join(img_root, str(class_idx).zfill(4))):
                os.renames(os.path.join(img_root, str(class_idx).zfill(4)),
                           os.path.join(img_root, str(class_idx).zfill(4) + "."
                                       + classname.replace(" ","_").replace("/","-")))
    with open(path_to_images, 'r') as in_images_file:
        with open(path_to_new_images, 'w') as out_images_file:
            for index, image_line in enumerate(in_images_file, 1):
                given_uuid, filename = image_line.strip('\n').split(' ', 1)
                class_index = filename[0:4]
                out_images_file.write(given_uuid + " " + filename.replace(class_index + "/",
                                                                          classes_dict[class_index] + "/") + "\n")


# Based on selective hierarchy choices merge certain classes together
def merge_based_on_hierarchy(images_file, images_merged_file, hierarchy_merge_file, in_dir_data, images_folder):
    path_to_images = os.path.join(in_dir_data, images_file)
    path_to_new_images = os.path.join(in_dir_data, images_merged_file)
    path_to_merges = os.path.join('.', hierarchy_merge_file)
    img_root = os.path.join(in_dir_data, images_folder)

    merge_dict = {}
    with open(path_to_merges, 'r') as path_to_merge_file:
        for merge_line in path_to_merge_file:
            original_index, new_index = merge_line.strip('\n').split(' ', 1)
            merge_dict[str(original_index).zfill(4)] = str(new_index).zfill(4)
    with open(path_to_images, 'r') as in_images_file:
        with open(path_to_new_images, 'w') as out_images_file:
            for index, image_line in enumerate(in_images_file, 1):
                given_uuid, filename = image_line.strip('\n').split(' ', 1)
                class_index = filename[0:4]
                if class_index in merge_dict:
                    found_lowest_rank = False
                    current_class_index = merge_dict[class_index]
                    while not found_lowest_rank:
                        if current_class_index in merge_dict:
                            current_class_index = merge_dict[current_class_index]
                        else:
                            found_lowest_rank = True
                    out_images_file.write(given_uuid + " " + filename.replace(class_index + "/",
                                                                              current_class_index + "/") + "\n")
                    os.renames(os.path.join(img_root, filename),
                               os.path.join(img_root, filename.replace(class_index + "/", current_class_index + "/")))
                else:
                    out_images_file.write(given_uuid + " " + filename + "\n")


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
    merge_based_on_hierarchy(IMAGES_FILE, IMAGES_MERGED_FILE, HIERARCHY_MERGE_FILE, IN_DIR_DATA, IMAGES_FOLDER)
    verify_images(IMAGES_MERGED_FILE, IN_DIR_DATA, IMAGES_FOLDER)
    rename_class_folders_to_class_names(CLASSES_FILE, CLASSES_RENAMED_TEMP_FILE, IMAGES_MERGED_FILE, IMAGES_RENAMED_TEMP_FILE,
                                        IN_DIR_DATA, IMAGES_FOLDER)
    remove_unwanted_species(SPLITS_FILE, IMAGES_RENAMED_TEMP_FILE, CLASSES_RENAMED_TEMP_FILE, SPLITS_REMOVAL_FILE,
                            IMAGES_REMOVAL_FILE, CLASSES_REMOVAL_FILE, UNWANTED_SPECIES_FILE,
                            IN_DIR_DATA, IMAGES_FOLDER)
    rename_class_folders_to_indexes(CLASSES_REMOVAL_FILE, CLASSES_RENAMED_FILE, IMAGES_REMOVAL_FILE,
                                    IMAGES_RENAMED_FILE, IN_DIR_DATA, IMAGES_FOLDER, CLASS_SHIFT)
    verify_images(IMAGES_RENAMED_FILE, IN_DIR_DATA, IMAGES_FOLDER)
