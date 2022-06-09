#!/bin/bash

cd uspppm

pip install -r requirements.txt
pip install bitsandbytes-cuda111

git config --global credential.helper store

git lfs install

DRIVE_DIR="/content/drive/My Drive/uspppm"

mkdir ~/.huggingface
cp "$DRIVE_DIR/hf.txt" ~/.huggingface/token

mkdir ~/.kaggle
mv "$DRIVE_DIR/kaggle.json" ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

mkdir data
kaggle competitions download -c us-patent-phrase-to-phrase-matching
unzip us-patent-phrase-to-phrase-matching.zip -d data

kaggle datasets download -d yasufuminakama/cpc-data
unzip -q cpc-data.zip -d data

cd "$DRIVE_DIR"
export WANDB_API_KEY=$( cat wandb.txt )
wandb login


export GIT_EMAIL=$( cat email.txt )
export GIT_NAME=$( cat name.txt )

cd /content/uspppm

git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"