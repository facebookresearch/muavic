#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# set variables
LANG=???
OUT_PATH=???
PRETRAINED_MODEL_PATH=???
LANG=en-fr
OUT_PATH=/private/home/anwarvic/muavic/muavic_internal/trained_models
PRETRAINED_MODEL_PATH=/private/home/anwarvic/tools/avhubert/large_vox_iter5.pt
NOISE_PATH=/checkpoint/anwarvic/data/musan-noise/all-for-training


# activate conda env
module load anaconda3/2022.05
module unload cuda cudnn
module load cuda/11.6 cudnn/v8.4.1.50-cuda.11.6
eval "$(conda shell.bash hook)"
conda activate avst


# set paths
ROOT=$(dirname "$(dirname "$(readlink -fm "$0")")")
ROOT=/private/home/anwarvic/muavic/muavic_internal
AV_HUBERT=${ROOT}/av_hubert/avhubert
MUAVIC=${ROOT}/muavic
# fix variables based on langauge
if [[ $LANG == *"-"* ]] ; then
    TASK="avst"
    IFS='-' read -r SRC TGT <<< ${LANG}
    DATA_PATH=${MUAVIC}/${SRC}/${TGT}
else
    TASK="avsr"
    TGT=${LANG}
    DATA_PATH=${MUAVIC}/${LANG}
fi

# create BPE dict/tokenizer using SentencePieceModel (SPM)
LABEL_DIR=${OUT_PATH}/${LANG}/labels
mkdir -p ${LABEL_DIR}
TOKENIZER_PATH=${LABEL_DIR}/tokenizer.model
if [[ ! -f ${TOKENIZER_PATH} ]]; then
    spm_train \
        --input=${DATA_PATH}/train.${TGT} \
        --model_prefix=${LABEL_DIR}/tokenizer \
        --vocab_size=1000 \
        --model_type=unigram \
        --character_coverage=1.0 \
        --bos_id=0 --pad_id=1 --eos_id=2 --unk_id=3
    
    # create fairseq dict
    awk 'FNR > 4 {print $(NF-1)" "1;}' ${LABEL_DIR}/tokenizer.vocab > ${LABEL_DIR}/dict.${TGT}.txt

    # copy labels from MuAViC
    cp ${DATA_PATH}/{train,valid,test}.${TGT} ${LABEL_DIR}/
fi

# start training
export PYTHONPATH="${ROOT}/av_hubert/fairseq:$PYTHONPATH"
fairseq-hydra-train \
    --config-dir ${AV_HUBERT}/conf/finetune \
    --config-name base_vox_433h \
        common.user_dir=${AV_HUBERT} \
        task.data=${DATA_PATH} \
        task.modalities=[audio,video] \
        task.label_dir=${LABEL_DIR} \
        task.labels=[${TGT}] \
        +task.min_sample_size=5 \
        task.max_sample_size=500 \
        task.tokenizer_bpe_model=${TOKENIZER_PATH} \
        dataset.train_subset=train \
        dataset.valid_subset=valid \
        model.w2v_path=${PRETRAINED_MODEL_PATH} \
        model.freeze_finetune_updates=4000 \
        model.dropout=0.1 \
        +model.no_pretrained_weights=false \
        optimization.lr=[0.0001] \
        optimization.max_update=30000 \
        lr_scheduler.warmup_steps=10000 \
        lr_scheduler.decay_steps=20000 \
        +task.noise_prob=0.25 \
        +task.noise_wav=${NOISE_PATH} \
        hydra.run.dir=${OUT_PATH}/${LANG}
