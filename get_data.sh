#!/bin/bash
rm -rf tulips1

URL='https://web.archive.org/web/20130509125535/http://mplab.ucsd.edu/wordpress/databases/tulips1.zip'
ZIP='tmp.zip'

wget $URL -O $ZIP && unzip $ZIP
rm $ZIP

rm -rf data
mkdir data

mv tulips1/PreprocessedData/MehreenSaeed/ContourParameters data/img
mv tulips1/RawData/tulips1.A/cepstrals data/audio

rm -rf tulips1
