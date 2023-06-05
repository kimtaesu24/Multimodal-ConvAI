#!/usr/bin/env sh

# For downloading models for SPICE
CORENLP=stanford-corenlp-full-2015-12-09
SPICELIB=pycocoevalcap/spice/lib
JAR=stanford-corenlp-3.6.0

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

if [ -f $SPICELIB/$JAR.jar ]; then
  echo "Found Stanford CoreNLP."
else
  echo "Downloading..."
  wget http://nlp.stanford.edu/software/$CORENLP.zip
  echo "Unzipping..."
  unzip $CORENLP.zip -d $SPICELIB/
  mv $SPICELIB/$CORENLP/$JAR.jar $SPICELIB/
  mv $SPICELIB/$CORENLP/$JAR-models.jar $SPICELIB/
  rm -f $CORENLP.zip
  rm -rf $SPICELIB/$CORENLP/
  echo "Done."
fi

# For downloading models for METEOR
PARAPHRASE_EN_URL="https://raw.githubusercontent.com/cmu-mtlab/meteor/master/data/paraphrase-en.gz"
METEOR_DATA_FNAME=pycocoevalcap/meteor/data/paraphrase-en.gz
if [ -f $METEOR_DATA_FNAME ];
then
    echo "Found paraphrase data for METEOR."
else
    echo "Downloading ${PARAPHRASE_EN_URL}"
    wget -O $METEOR_DATA_FNAME $PARAPHRASE_EN_URL
    echo "Done."
fi
