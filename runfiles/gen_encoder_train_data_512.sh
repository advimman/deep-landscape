#!/usr/bin/env bash

BASEDIR="$(dirname $0)"

"$BASEDIR/gen_encoder_train_data.sh" 512 445000 --scale-images 256
