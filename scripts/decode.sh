#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

LANG=???
GROUP=???
MODALITIES=???
MODEL_PATH=???
TOKENIZER_PATH=???
OUT_PATH=???


# set paths
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
AV_HUBERT=${ROOT}/av_hubert/avhubert
MUAVIC=${ROOT}/muavic


# fix variables based on langauge
if [[ $LANG == *"-"* ]] ; then
    TASK="avst"
    IFS='-' read -r SRC TGT <<< ${LANG}
    USE_BLEU=true
    DATA_PATH=${MUAVIC}/${SRC}/${TGT}
else
    TASK="avsr"
    TGT=${LANG}
    USE_BLEU=false
    DATA_PATH=${MUAVIC}/${LANG}
fi

# start decoding
python -B ${AV_HUBERT}/infer_s2s.py \
    --config-dir ${AV_HUBERT}/conf \
    --config-name s2s_decode \
        common.user_dir=${AV_HUBERT} \
        override.modalities=[${MODALITIES}] \
        dataset.gen_subset=${GROUP} \
        override.data=${DATA_PATH} \
        override.label_dir=${DATA_PATH} \
        override.tokenizer_bpe_model=${TOKENIZER_PATH} \
        generation.beam=5 \
        generation.lenpen=0 \
        dataset.max_tokens=3000 \
        override.labels=[${TGT}] \
        override.eval_bleu=${USE_BLEU} \
        common_eval.path=${MODEL_PATH} \
        common_eval.results_path=${OUT_PATH}/${LANG}_${TASK}