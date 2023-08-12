# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
import dlib
import torch
import logging
import numpy as np
from ffmpy import FFmpeg
from pathlib import Path
from scipy.io import wavfile
from collections import OrderedDict, defaultdict

from utils import (
    load_meanface_metadata,
    download_extract_file_if_not,
)
from fairseq import utils, checkpoint_utils
from fairseq.dataclass.configs import GenerationConfig


logger = logging.getLogger(__name__)

TIME_TRACKER = OrderedDict() # to track time-taken per step
MAX_MISSING_FRAMES_RATIO = 0.75 #max video frames that is ok to be missing
USE_CUDA = torch.cuda.is_available()


def load_noise_samples(noise_path):
    download_extract_file_if_not(
        url="https://dl.fbaipublicfiles.com/muavic/noise_samples.tgz",
        compressed_filepath= (noise_path/"noise_samples.tgz"),
        download_filename="noise_samples"
    )
    noise_dict = defaultdict(list)
    for wav_filepath in (noise_path/"noise_samples").rglob('*.wav'):
        category = wav_filepath.parent.stem
        noise_dict[category].append(str(wav_filepath))
    return noise_dict


def load_av_models(av_models_path):
    av_resources = defaultdict(dict)
    # Load AV-HuBERT Models
    for parent_path in sorted(av_models_path.glob("*")): # SORT THEM
        if parent_path.is_file(): continue # parent_path has to be a directory
        # prepare needed variables
        key = parent_path.stem
        lang_label = key.split('_')[0].split('-')[-1]
        label_path = str(parent_path)
        ckpt_path = str(parent_path / "checkpoint_best.pt")
        # load av model
        arg_overrides = {
            "modalities": ["audio", "video"],
            "data": label_path,
            "labels": [lang_label],
            "label_dir": label_path,
            "tokenizer_bpe_model": f"{label_path}/tokenizer.model",
            "noise_prob": 0,
            "noise_wav": None,
        }
        models, _, task = checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path], arg_overrides
        )
        models = [
            model.eval().cuda() if USE_CUDA else model.eval()
            for model in models
        ]
        generator = task.build_generator(models, GenerationConfig(beam=1))
        # save info
        av_resources[key]["model"] = models
        av_resources[key]["task"] = task
        av_resources[key]["generator"] = generator
        del models, task, generator
    return av_resources


def add_noise(signal, noise, snr):
    """
    signal: 1D tensor in [-32768, 32767] (16-bit depth)
    noise: 1D tensor in [-32768, 32767] (16-bit depth)
    snr: tuple or float
    """
    signal = signal.astype(np.float32)
    noise = noise.astype(np.float32)

    if type(snr) == tuple:
        assert len(snr) == 2
        snr = np.random.uniform(snr[0], snr[1])
    else:
        snr = float(snr)

    if len(signal) > len(noise):
        ratio = int(np.ceil(len(signal) / len(noise)))
        noise = np.concatenate([noise for _ in range(ratio)])
    if len(signal) < len(noise):
        start = 0
        noise = noise[start : start + len(signal)]

    amp_s = np.sqrt(np.mean(np.square(signal), axis=-1))
    amp_n = np.sqrt(np.mean(np.square(noise), axis=-1))
    noise = noise * (amp_s / amp_n) / (10 ** (snr / 20))
    mixed = signal + noise

    # Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
            reduction_rate = max_int16 / mixed.max(axis=0)
        else:
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    mixed = mixed.astype(np.int16)
    return mixed
 

def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None ]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(
                landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
            )
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[: valid_frames_idx[0]] = [
            landmarks[valid_frames_idx[0]]
        ] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
            len(landmarks) - valid_frames_idx[-1]
        )
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks
 
 
def track_time(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        TIME_TRACKER[f"{func.__name__!r}"] = round(t2-t1, 2)
        return result
    return wrap_func


def load_needed_models_for_lip_movement(metadata_path):
    metadata_path.mkdir(parents=True, exist_ok=True)
    # load dlib's face detector (NOTE: can be replaced by any other model)
    logger.debug("Loading frontal face detector!")
    detector = dlib.get_frontal_face_detector()
    # load landmark predictor
    logger.debug("Loading shape predictor!")
    filename = "shape_predictor_68_face_landmarks.dat"
    download_extract_file_if_not(
        url=f"http://dlib.net/files/{filename}.bz2",
        compressed_filepath=metadata_path/f"{filename}.bz2",
        download_filename=filename
    )
    shape_predictor_path = metadata_path / filename
    predictor = dlib.shape_predictor(str(shape_predictor_path))
    # load metadata
    logger.debug("Loading mean-face metadata!")
    mean_face_landmarks = load_meanface_metadata(metadata_path)
    return (
        detector, predictor, mean_face_landmarks
    )


@track_time
def mix_audio_with_noise(webcam_video, audio_file, out_file, noise_wav_file, snr):
    # get audio from webcam video
    FFmpeg(
        inputs={webcam_video: None},
        outputs={audio_file: "-v quiet -vn -acodec pcm_s16le -ar 16000 -ac 1"},
    ).run()
    # read audio WAV file
    sr, audio = wavfile.read(audio_file)
    # read noise WAV file
    logger.debug(f"Noise Wav used is {noise_wav_file}")
    _, noise_wav = wavfile.read(noise_wav_file)
    # mix audio + noise
    mixed = add_noise(audio, noise_wav, snr)
    # save resulting noisy audio WAV file
    wavfile.write(out_file, sr, mixed)
    return mixed


@track_time
def infer_av_hubert(
    av_models,
    av_task,
    av_generator,
    vid_filepath,
    audio_filepath,
    duration
):
    def decode_fn(x, av_task, gen, gen_subset_name):
        dictionary = av_task.target_dictionary
        symbols_ignore = gen.symbols_to_strip_from_output
        symbols_ignore.add(dictionary.pad())
        return av_task.datasets[gen_subset_name].label_processors[0].decode(
            x, symbols_ignore
        )
    logger.debug("Preparing manifest & label files.")
    gen_subset = "test"
    av_label_path = av_task.cfg.label_dir
    av_label_ext = av_task.cfg.labels[0]
    manifest_filepath = Path(av_label_path) / f"{gen_subset}.tsv"
    a_frames = int(duration * 16000) # NOTE: 16000 is the audio sample rate
    v_frames = int(25 * duration) #NOTE: 25 is the video framerate per second
    with open(manifest_filepath, "w") as fout:
        fout.write("/\n")
        fout.write(
            f"id\t{vid_filepath}\t{audio_filepath}\t{v_frames}\t{a_frames}\n"
        )
    label_filepath = f"{av_label_path}/{gen_subset}.{av_label_ext}"
    with open(label_filepath, "w") as fo:
        fo.write("[PLACEHOLDER]\n")
    logger.debug(f"Manifest filepath: {manifest_filepath}")
    logger.debug(f"Label filepath: {label_filepath}")

    av_task.load_dataset(gen_subset, task_cfg=av_task.cfg)
    itr = av_task.get_batch_iterator(
        dataset=av_task.dataset(gen_subset)
    ).next_epoch_itr(shuffle=False)
    sample = next(itr)
    if USE_CUDA: sample = utils.move_to_cuda(sample)
    hypos = av_task.inference_step(av_generator, av_models, sample)
    # ref = decode_fn(sample['target'][0].int().cpu())
    hypo = hypos[0][0]['tokens'].int().cpu()
    hypo = decode_fn(hypo, av_task, av_generator, gen_subset)
    # clear un-needed files
    os.remove(label_filepath)
    os.remove(manifest_filepath)
    logger.debug("Done cleaning un-needed files")
    return hypo
