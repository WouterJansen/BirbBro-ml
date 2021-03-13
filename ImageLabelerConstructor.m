%% Parameters

dataFolder = "data";
imagesFilename = "images.txt";
trainSplitFilename = "train_test_split.txt";
imagesFolder = "images";

%% load image UUIDs and their location paths

fileID = fopen(dataFolder + "/" + imagesFilename);
file = textscan(fileID, '%s %s','Delimiter',' ');
fclose(fileID);
imageUUIDs = string(file{1});
imageLocations = string(file{2});

%% Find the images that require labeling based on the training list

fileID = fopen(dataFolder + "/" + trainSplitFilename);
file = textscan(fid, '%s %s', 'Delimiter', ' ');
fclose(fileID);
imageTrainingUUIDs = string(file{1});
imageTrainingToggle = ~str2num(cell2mat(file{2}));
imageTrainingUUIDs(imageTrainingToggle) = [];

[sharedvals, found_indexes] = intersect(imageUUIDs, imageTrainingUUIDs, 'stable');
imageUUIDsToLabel = imageUUIDs(found_indexes);
imageLocationsToLabel = imageLocations(found_indexes);

%% Copy all images to label to temporary folder

mkdir(dataFolder + "/" + imagesFolder + "_label");
for image_idx = 1 : size(imageLocationsToLabel, 1)
    copyfile(dataFolder + "/" + imagesFolder + "/" + imageLocationsToLabel(image_idx), dataFolder + "/" + imagesFolder + "_label" + "/" + imageUUIDsToLabel(image_idx) + ".jpg");
end

%% Create image labeler

imageLabeler(dataFolder + "/" + imagesFolder + "_label")