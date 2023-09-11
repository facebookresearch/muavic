MuAViC
======

A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation.

[Paper](https://www.isca-speech.org/archive/pdfs/interspeech_2023/anwar23_interspeech.pdf)

# Overview

MuAViC provides
- 1200 hours of transcribed audio-visual speech for 9 languages (English, Arabic, German, Greek,
Spanish, French, Italian, Portuguese and Russian)
- text translations for 6 English-to-X directions and 6 X-to-English directions (X = Greek,
Spanish, French, Italian, Portuguese or Russian)

<div align="center">
    <img src="https://github.com/facebookresearch/muavic/assets/15960959/ef144a70-3def-4742-9469-4d5c505502d0" alt="MuAViC data statistics" width=560>
</div>


The raw data is collected from TED/TEDx talk recordings.

#### Detailed statistics

<details><summary>Audio-Visual Speech Recognition</summary><p>

| Language | Code | Train Hours (H+P) | Train Speakers |
|:---:|:---:|:---:|:---:|
| English | En |  436 + 0 | 4.7K |
| Arabic | Ar |  16 + 0 | 95 |
| German | De |  10 + 0 | 53 |
| Greek | El | 25 + 0 | 113 |
| Spanish | Es | 178 + 0 | 987 |
| French | Fr |  176 + 0 | 948 |
| Italian | It |  101 + 0 | 487 |
| Portuguese | Pt | 153 + 0 | 810 |
| Russian | Ru | 49 + 0 | 238 |

</p></details>


<details><summary>Audio-Visual En-X Speech-to-Text Translation</summary><p>

| Direction | Code | Train Hours (H+P) | Train Speakers |
|:---:|:---:|:---:|:---:|
| English-Greek | En-El | 17 + 420 | 4.7K |
| English-Spanish | En-Es | 21 + 416 | 4.7K |
| English-French | En-Fr |  21 + 416 | 4.7K |
| English-Italian | En-It |  20 + 417 | 4.7K |
| English-Portuguese | En-Pt | 18 + 419 | 4.7K |
| English-Russian | En-Ru | 20 + 417 | 4.7K |

</p></details>


<details><summary>Audio-Visual X-En Speech-to-Text Translation</summary><p>

| Direction | Code | Train Hours (H+P) | Train Speakers |
|:---:|:---:|:---:|:---:|
| Greek-English | El-En |  8 + 17 | 113 |
| Spanish-English | Es-En |  64 + 114 | 987 |
| French-English | Fr-En |  45 + 131 | 948 |
| Italian-English | It-En |  48 + 53 | 487 |
| Portuguese-English | Pt-En | 53 + 100 | 810 |
| Russian-English | Ru-En | 8 + 41 | 238 |

</p></details>


# Getting Data

We provide scripts to generate the audio/video data and AV-HuBERT training manifests for MuAViC.

First, clone this repo for the scripts
```bash
git clone https://github.com/facebookresearch/muavic.git
```

Install required packages:
```bash
conda install -c conda-forge ffmpeg==4.2.2
conda install -c conda-forge sox
pip install -r requirements.txt
```

Then get audio-visual speech recognition and translation data via
```bash
python get_data.py --root-path ${ROOT} --src-lang ${SRC_LANG}
```
where the speech language `${SRC_LANG}` is one of `en`, `ar`, `de`, `el`, `es`, `fr`, `it`, `pt` and `ru`.

Generated data will be saved to `${ROOT}/muavic`:
- `${ROOT}/muavic/${SRC_LANG}/audio` for processed audio files
- `${ROOT}/muavic/${SRC_LANG}/video` for processed video files
- `${ROOT}/muavic/${SRC_LANG}/*.tsv` for AV-HuBERT AVSR training manifests
- `${ROOT}/muavic/${SRC_LANG}/${TGT_LANG}/*.tsv` for AV-HuBERT AVST training manifests


# Models

In the following table, we provide all end-to-end trained models mentioned
in our paper:

<table align="center">
    <tr>
        <th>Task</th>
        <th>Languages</th>
        <th>Best Checkpoint</th>
        <th>Dictionary</th>
        <th>Tokenizer</th>
    </tr>
    <tr>
        <td rowspan="10">AVSR</td>
        <th>ar</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/ar_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/ar_avsr/dict.ar.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/ar_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>de</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/de_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/de_avsr/dict.de.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/de_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>el</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/el_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/el_avsr/dict.el.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/el_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>en</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en_avsr/dict.en.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>es</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/es_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/es_avsr/dict.es.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/es_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>fr</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/fr_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/fr_avsr/dict.fr.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/fr_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>it</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/it_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/it_avsr/dict.it.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/it_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>pt</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/pt_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/pt_avsr/dict.pt.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/pt_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>ru</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/ru_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/ru_avsr/dict.ru.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/ru_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>ar,de,el,es,fr,it,pt,ru</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/x_avsr/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/x_avsr/dict.x.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/x_avsr/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <td rowspan="13">AVST</td>
        <th>en-el</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-el_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-el_avst/dict.el.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-el_avst/tokenizer.model">tokenizer</a></th>
    </tr>
     <tr>
        <th>en-es</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-es_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-es_avst/dict.es.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-es_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>en-fr</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-fr_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-fr_avst/dict.fr.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-fr_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>en-it</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-it_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-it_avst/dict.it.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-it_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>en-pt</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-pt_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-pt_avst/dict.pt.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-pt_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>en-ru</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-ru_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-ru_avst/dict.ru.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/en-ru_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>el-en</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/el-en_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/el-en_avst/dict.en.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/el-en_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>es-en</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/es-en_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/es-en_avst/dict.en.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/es-en_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>fr-en</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/fr-en_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/fr-en_avst/dict.en.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/fr-en_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>it-en</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/it-en_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/it-en_avst/dict.en.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/it-en_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>pt-en</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/pt-en_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/pt-en_avst/dict.en.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/pt-en_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>ru-en</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/ru-en_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/ru-en_avst/dict.en.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/ru-en_avst/tokenizer.model">tokenizer</a></th>
    </tr>
    <tr>
        <th>{el,es,fr,it,pt,ru}-en</th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/x-en_avst/checkpoint_best.pt">best_ckpt.pt</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/x-en_avst/dict.en.txt">dict</a></th>
        <th><a href="https://dl.fbaipublicfiles.com/muavic/models/x-en_avst/tokenizer.model">tokenizer</a></th>
    </tr>
</table>


# Demo

To try out our state-of-the-art audio-visual models with different
audio and video inputs, including a recorded video through the webcam or 
an uploaded video, checkout our demo:



https://github.com/facebookresearch/muavic/assets/15960959/d03df3b0-488c-443c-ba3b-452b1a5765d8



You can read more about our model in the [README](https://github.com/facebookresearch/muavic/tree/main/demo) file in the demo folder.
# Training

For training Audio-Visual models, we are going to use
[AV-HuBERT](https://github.com/facebookresearch/av_hubert) framework.

1. Clone and install AV-HuBERT in the root directory:
    ```bash
    $ # Clone the "muavic" branch of av_hubert's repo
    $ git -b muavic clone https://github.com/facebookresearch/av_hubert.git
    $ # Set the fairseq version
    $ cd avhubert
    $ git submodule init
    $ git submodule update
    $ # Install av-hubert's requirements
    $ pip install -r requirements.txt
    $ # Install fairseq
    $ cd fairseq
    $ pip install --editable ./
    ```

2. Download an AV-HuBERT pre-trained model from [here](http://facebookresearch.github.io/av_hubert).

3. Open the training script ([`scripts/train.sh`](https://github.com/facebookresearch/muavic/tree/main/scripts/train.sh)) and replace these variables:
    ```bash
    # language direction (e.g "en" or "en-fr")
    LANG=

    # path where output trained models will be located
    OUT_PATH= 

    # path to the downloaded pre-trained model
    PRETRAINED_MODEL_PATH=
    ```

4. Run the training script:
    ```bash
    $ bash scripts/train.sh
    ```

> Note:\
All audio-visual models found [here](https://github.com/facebookresearch/muavic#models)
used the `large_vox_iter5.pt` pre-trained model.


# Decoding/Evaluating

To evaluate your trained model (or our [trained models]()) against MuAViC, follow
these steps:

1. Open the decoding script ([`scripts/decode.sh`](https://github.com/facebookresearch/muavic/tree/main/scripts/decode.sh)) and replace these variables:
    ```bash
    # language direction (e.g "en" or "en-fr")
    LANG=???

    # data split (e.g "test" or "valid")
    GROUP=???

    # inference modality (choices: "audio", "video", "audio,video")
    MODALITIES=???

    # path to the trained model
    MODEL_PATH=???

    # path where decoding results and scores will be located
    OUT_PATH=???
    ```

2. Run the decoding script:
    ```bash
    $ bash scripts/decode.sh
    ```
# License

CC-BY-NC 4.0


# Citation

```bibtex
@article{anwar2023muavic,
  title={MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation},
  author={Anwar, Mohamed and Shi, Bowen and Goswami, Vedanuj and Hsu, Wei-Ning and Pino, Juan and Wang, Changhan},
  journal={arXiv preprint arXiv:2303.00628},
  year={2023}
}
```
