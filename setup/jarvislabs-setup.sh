#!/bin/bash

cd uspppm

pip install -r requirements.txt
pip install bitsandbytes-cuda113

git config --global credential.helper store

apt-get install git-lfs
git lfs install

FILE="hf.txt"
if [ -e $FILE ]
then
    echo "File $FILE exists"
else
    read -s -p "Upload hf.txt: " ignore
fi
mkdir ~/.huggingface
cp hf.txt ~/.huggingface/token

FILE="~/.kaggle/kaggle.json"
if [ -e $FILE ]
then
    echo "File $FILE exists"
else
    mkdir ~/.kaggle
    read -s -p "Upload kaggle.json: " ignore
    mv kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

    mkdir data
    kaggle kernels output nbroad/uspppm-folds -p data
    kaggle kernels output nbroad/cpc-titles-explore -p data
fi

FILE="wandb.txt"
if [ -e $FILE ]
then
    echo "File $FILE exists"
else
    read -s -p "Upload wandb.txt: " ignore
fi

export WANDB_API_KEY=$( cat wandb.txt )
wandb login

export GIT_EMAIL=$( cat email.txt )
export GIT_NAME=$( cat name.txt )

git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"