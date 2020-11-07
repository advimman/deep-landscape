#!/usr/bin/env bash

BASEDIR="$(dirname $0)"
BASEDIR="$(realpath $BASEDIR)"

CONFIG_PATH="$BASEDIR/../configs/encoders/512_resnet.yaml"

"$BASEDIR/../inference/train_encoder.py" "$CONFIG_PATH"
