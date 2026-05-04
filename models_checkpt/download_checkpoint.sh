#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoint_02-05-26"

mkdir -p "$CHECKPOINT_DIR"

echo "Downloading checkpoint..."

wget -O "$CHECKPOINT_DIR/best_model.pt" \
"https://drive.google.com/uc?export=download&id=1tpZLx2p515iHDeYOocyt2IG5UqTdRFHM"

echo "Done: $CHECKPOINT_DIR/best_model.pt"
