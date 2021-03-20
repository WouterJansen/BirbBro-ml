
# BirdBro-ml

BirdBro is an Android application to receive notifications of your friendly bird friends visiting your birdfeeder.
This app is connected over Google Firebase cloud to a internet-connected ESP32 camera system inside the birdfeeder to capture images and upload them to the app. 
A trained ML model is used to predict the bird class. This repository holds some of the scripts used to gather the data, train it, test it and deploy it.

You can find more information on the ESP32 camera system here: https://github.com/WouterJansen/BirbBro-esp32

You can find more information on the Android app here: https://github.com/WouterJansen/BirbBro-app


## Usage

All scripts have their parameters defined at the top of each script.

###  Gathering images
[A parser script](flickr_parser/flickr_class_images_scraper.py) is used to scrape Flickr for specific key terms (bird species) and download the images to specific class folders.
It automatically will make the typical files used for creating ML datasets such as a classes list, an images list and a random split into training and none-training images.
[A script](flickr_parser/remove_images_not_found.py) is also available that after you might manually remove some images that you didn't like, to also remove them from all the definition text files.
Lastly, [a script](flickr_parser/resize_images.py) is there to resize all images to a desired size with black bars to facilitate training and testing. 

###  Training & testing
In the main folder the [main training script](training.py) is used using PyTorch to train based on a pretrained ResNet-50 model.
[A test script](test.py) is also available to randomly take some images from the test pool and plot them while showing the real and predicted class together with the prediction certainty. 

###  Convert for Android
[A conversion script](convert_to_android.py) is available to convert the PyTorch model to a format that can be used within an Android app. 

###  Generate bounding boxes
While unused by the BirdBro training and test scripts in the end, [a Matlab script](ImageLabelerConstructor.m) was made to facilitate creating an Imagelabeler dataset easily. This then allows you to manually label the images used for training.
[A secondary Matlab script](LabelsToBBoxesFile.m) is available to then convert this groundtruth to a text file with all bounding box information.

###  Public dataset conversion
While unused by the BirdBro training and test scripts in the end, several attempts were made to finetune some public available datasets of bird images to the right format used in the training and test scripts.
These are available in [a seperate folder](public_dataset_conversion_scripts).


## Sources

  - Most of the network follows the structure from here: https://github.com/slipnitskaya/caltech-birds-advanced-classification
  - Flickr scrapr inspired by: https://github.com/ultralytics/flickr_scraper
  - Converting and running a PyTorch model on Android: https://heartbeat.fritz.ai/pytorch-mobile-image-classification-on-android-5c0cfb774c5b?gi=edc55b03afef