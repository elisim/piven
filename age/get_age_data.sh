#!/bin/sh

DS="kmader/rsna-bone-age"
ZIP_NAME="rsna-bone-age.zip"
OUT_FOLDER='./rsna-bone-age'

kaggle datasets download -d $DS
unzip -q $ZIP_NAME -d $OUT_FOLDER
rm -vf $ZIP_NAME
cp 'Bone age ground truth.xlsx' $OUT_FOLDER