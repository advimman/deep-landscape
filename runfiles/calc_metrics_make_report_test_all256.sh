#!/usr/bin/env bash

BASEDIR="$(dirname $0)"
BASEDIR="$(realpath $BASEDIR)"

for CONFIG_NAME in 01_eoifs 02_eoif 03_eoi 04_eo 05_e 06_mo 07_i2s
do
    INPATH="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encode_and_animate_results/test_images/256/$CONFIG_NAME/"
    OUTPATH="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encode_and_animate_results/test_images/256/$CONFIG_NAME/metrics_final.tsv"

    "$BASEDIR/../inference/calc_metrics_video.py" "$INPATH" "$OUTPATH"
done

CONFIG_NAME=test_videos
INPATH="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encode_and_animate_results/test_images/256/$CONFIG_NAME/"
OUTPATH="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encode_and_animate_results/test_images/256/$CONFIG_NAME/metrics_final.tsv"

"$BASEDIR/../inference/calc_metrics_video.py" "$INPATH" "$OUTPATH" --thisisreal --frametemplate "{}.jpg"

INPATH="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encode_and_animate_results/test_images/256/*"
OUTPATH="${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encode_and_animate_results/test_images/256_report/"
"$BASEDIR/../inference/make_report.py" "$INPATH" "$OUTPATH" --nost
