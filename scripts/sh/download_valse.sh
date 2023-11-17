#!/bin/zsh

IMG_URL="http://vision.stanford.edu/yukezhu/visual7w_images.zip"
VALSE_URL="https://raw.githubusercontent.com/Heidelberg-NLP/VALSE/main/data/existence.json"

IMG_FOLDER_PATH="../../data/raw/visual7w"
VALSE_FOLDER_PATH="../../data/raw/valse"
VALSE_DESTINATION_DIRECTORY="$VALSE_FOLDER_PATH/valse_existence.json"

wget $IMG_URL -O v7w_images.zip
mkdir -p $IMG_FOLDER_PATH
unzip v7w_images.zip -d $IMG_FOLDER_PATH
rm v7w_images.zip

wget $VALSE_URL -O valse_existence.json
mkdir -p $VALSE_FOLDER_PATH
mv valse_existence.json $VALSE_DESTINATION_DIRECTORY
