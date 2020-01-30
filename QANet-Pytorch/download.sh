#!/usr/bin/env bash

# Install dependencies
sudo apt-get -y install unzip wget
sudo pip3.6 install spacy==2.0.11

# Download SQuAD
SQUAD_DIR=./data/original/SQuAD
mkdir -p $SQUAD_DIR
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O $SQUAD_DIR/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O $SQUAD_DIR/dev-v1.1.json

# Download GloVe
GLOVE_DIR=./data/original/Glove
mkdir -p $GLOVE_DIR
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O $GLOVE_DIR/glove.840B.300d.zip
unzip $GLOVE_DIR/glove.840B.300d.zip -d $GLOVE_DIR

PROCESSED_DIR=./data/processed/SQuAD
mkdir -p $PROCESSED_DIR

# Download Spacy language models
sudo python3.6 -m spacy download en
