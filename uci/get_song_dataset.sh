#!/bin/sh
URL="http://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
ZIP_NAME="YearPredictionMSD.txt.zip"

wget $URL
unzip $ZIP_NAME -d "UCI_Datasets"
rm -vf $ZIP_NAME
