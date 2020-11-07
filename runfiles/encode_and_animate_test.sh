#!/usr/bin/env bash

if (($# < 2))
then
    echo "Usage: $0 config_name homography_dir"
    exit 1
fi

BASEDIR="$(dirname $0)"
BASEDIR="$(realpath $BASEDIR)"

CONFIG_NAME=$1
CONFIG_PATH="$BASEDIR/../configs/inference/$CONFIG_NAME.yaml"
HOMOGRAPHY_DIR=$2

INGLOB="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/test_images/*.jpg"
OUTPATH="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encode_and_animate_results/test_images/$CONFIG_NAME"

"$BASEDIR/../inference/encode_and_animate.py" "$CONFIG_PATH" "$INGLOB" "$OUTPATH" $HOMOGRAPHY_DIR
