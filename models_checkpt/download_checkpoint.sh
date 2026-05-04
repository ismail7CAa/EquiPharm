#!/bin/bash

mkdir -p checkpoint_02-05-26

echo "Downloading checkpoint..."

wget -O checkpoint_02-05-26/best_model.pt \
"https://drive.google.com/uc?export=download&id=1tpZLx2p515iHDeYOocyt2IG5UqTdRFHM"

echo "Done."
