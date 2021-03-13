%% Parameters

dataFolder = "data";
labelsFile = "labels.mat";
bboxFilename = "bboxes.txt";

%% Load the groundtruth labels

load(labelsFile)
imageLocations = split(string(gTruth.DataSource.Source),"\");
imageLocations = split(imageLocations(:, end), ".");
imageUUIDs = imageLocations(:, 1);
imageBoundingBoxes = cell2mat(table2array(gTruth.LabelData));

%% Write all bounding box information to the text file by UUID

fileID = fopen(dataFolder + "/" + bboxFilename, 'w');
for image_index = 1 : size(imageBoundingBoxes, 1)
    fprintf(fileID, '%s %i %i %i %i\n', imageUUIDs(image_index), imageBoundingBoxes(image_index, 1), imageBoundingBoxes(image_index, 2), imageBoundingBoxes(image_index, 3), imageBoundingBoxes(image_index, 4));
end
fclose(fileID);