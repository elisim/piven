#!/bin/sh
URL="https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar"
TAR_NAME="imdb_crop.tar"

wget $URL
tar -xf $TAR_NAME
rm -vf $TAR_NAME