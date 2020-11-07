#!/usr/bin/env bash

BASEDIR="$(dirname $0)"
BASEDIR="$(realpath $BASEDIR)"
HOMOGRAPHY_DIR=$1

INGLOB="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/test_images/*.jpg"

for CONFIG_NAME in 01_eoifs 02_eoif 03_eoi 04_eo 05_e 06_mo 07_i2s
do
    CONFIG_PATH="$BASEDIR/../configs/inference/512/$CONFIG_NAME.yaml"
    OUTPATH="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encode_and_animate_results/test_images/512/$CONFIG_NAME"

    "$BASEDIR/../inference/encode_and_animate.py" "$CONFIG_PATH" "$INGLOB" "$OUTPATH" $HOMOGRAPHY_DIR
done
