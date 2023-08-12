# AVSR/AVST Demo

This is a demonstration app built to showcase the power of audio-visual speech
model for speech recognition and translations tasks. In this demo app, you can
test the performance of our state-of-the-art audio-visual models on different
audio and video inputs, including recorded video through the webcam or 
by uploading real-world videos with different backgrounds and noise levels.

Whether you are a researcher, developer, or just curious about the latest
advances in speech recognition, this demo app provides a unique opportunity
to explore the power of audio visual models in a user-friendly and interactive
way.

https://github.com/facebookresearch/muavic/assets/15960959/d03df3b0-488c-443c-ba3b-452b1a5765d8

> Note:\
All models were trained using Meta's
[AV-HuBERT](https://github.com/facebookresearch/av_hubert) framework.

## Getting Started

In order to run this demo app on your machine, follow these steps:

- Install needed dependencies
    ```bash
    $ pwd
    ~/muavic/demo
    
    $ pip install -r requirements.txt
    ```

- Download trained AV models, full list can be found
[here](https://github.com/facebookresearch/muavic#models):
    ```bash
    $ # define needed variables
    $ TASK="avst"
    $ LANG="en-fr"
    $ URL="https://dl.fbaipublicfiles.com/muavic/models/${LANG}_${TASK}"
    $
    $ #download model checkpoint
    $ wget ${URL}/checkpoint_best.pt
    $
    $ #download dict
    $ wget ${URL}/dict."${LANG#*-}".txt
    $
    $ #download tokenizer
    $ wget ${URL}/tokenizer.model
    ```

Now, you can run the demo:
```bash
$ pwd
~/muavic/demo
$ python run_demo.py
    --avhubert-path ${AVHUBERT} \
    --av-models-path ${AV_MODELS} \
    --output-path ${OUT_PATH}
```
Such as that:

- `AVHUBERT`: path to the cloned `av_hubert/avhuert`.
- `OUT_PATH`: path where all output files will be saved.
- `AV_MODELS`: path where pre-trained models are downloaded, it follows this
    structure:
    ```text
    .
    ├── ar_avsr
    │   ├── checkpoint_best.pt
    │   ├── dict.ar.txt
    │   └── tokenizer.model
    ├── en-es_avst
    │   ├── checkpoint_best.pt
    │   ├── dict.es.txt
    │   └── tokenizer.model
    └── ...
    ```

After running this command successfully, you can access tha app at this
url: http://127.0.0.1:7860/
