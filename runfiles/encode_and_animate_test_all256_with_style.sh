#!/usr/bin/env bash

BASEDIR="$(dirname $0)"
BASEDIR="$(realpath $BASEDIR)"
HOMOGRAPHY_DIR=$1

INGLOB="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/test_images/*.jpg"

for CONFIG_NAME in style_eoif style_eoifs
do
    CONFIG_PATH="$BASEDIR/../configs/inference/256/$CONFIG_NAME.yaml"
    OUTPATH="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encode_and_animate_results/test_images/256_style/$CONFIG_NAME"

    "$BASEDIR/../inference/encode_and_animate.py" "$CONFIG_PATH" "$INGLOB" "$OUTPATH" $HOMOGRAPHY_DIR
done
