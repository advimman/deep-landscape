#!/usr/bin/env bash

BASEDIR="$(dirname $0)"
BASEDIR="$(realpath $BASEDIR)"

CONFIG_PATH="$BASEDIR/../configs/encoders/256_mlp.yaml"

"$BASEDIR/../inference/mlp_approximation.py" "$CONFIG_PATH"
