#!/usr/bin/env bash


if (($# < 2))
then
    echo "Usage: $0 model_name iteration"
    exit 1
fi


BASEDIR=$(dirname $0)

MODEL_NAME=$1
ITER=$2

shift
shift

"$BASEDIR/../inference/generate_dataset.py" --num-samples 200000 --z-scale 3 --trunc 1 \
    --model-iter $ITER --batch-size 6 --mixin-prob 0.9 --scale-images 256 \
    "$MODEL_NAME" \
    "${DEEP_LANDSCAPE_RESULTS_DIR:-results}/encoder_train_dataset/${MODEL_NAME}_${ITER}" \
    $@
