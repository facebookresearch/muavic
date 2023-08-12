# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import sys
import time
import random
import shutil
import hashlib
import logging
import argparse
import gradio as gr
from tqdm import tqdm
from pathlib import Path
from ffmpy import FFmpeg

sys.path.insert(0, str(Path(__file__).parent.parent))
from demo_utils import *
from utils import (
    split_video_to_frames,
    resize_frames,
    crop_patch,
    save_video,
)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def detect_landmark(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = DETECTOR(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = PREDICTOR(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


@track_time
def extract_lip_movement(
        webcam_video,
        in_video_filepath,
        out_lip_filepath,
        num_workers
    ):
    def copy_video_if_ready(webcam_video, out_path):
        with open(webcam_video, 'rb') as fin:
            curr_md5hash = hashlib.md5(fin.read()).hexdigest()
        # check if the current hash matches anything in the cache
        if curr_md5hash in VIDEOS_CACHE:
            dst_path = VIDEOS_CACHE[curr_md5hash]
            # copy needed files
            shutil.copy(dst_path/"video.mp4", out_path)
            shutil.copy(dst_path/"lip_movement.mp4", out_path)
            shutil.copy(dst_path/"raw_video.md5", out_path)
            return True
        else:
            VIDEOS_CACHE[curr_md5hash] = out_path
            with open(out_path/"raw_video.md5", 'w') as fout:
                fout.write(curr_md5hash)
        return False

    if copy_video_if_ready(webcam_video, in_video_filepath.parent):
        logger.info("Skip video processing; Loading the cached one!!")
        return
    # change video framerate to 25 and lower resolution for faster processing
    logger.info("Adjust video framerate to 25")
    FFmpeg(
        inputs={webcam_video: None},
        outputs={in_video_filepath: "-v quiet -filter:v fps=fps=25 -vf scale=640:480"},
    ).run()
    # convert video to a list of frames
    logger.info("Converting video into frames")
    frames = list(split_video_to_frames(in_video_filepath))
    
    # Get face landmarks from video 
    logger.info("Extract face landmarks from video frames")
    landmarks = [
        detect_landmark(frame)
        for frame in tqdm(frames, desc="Detecting Lip Movement")
    ]
    # landmarks = process_map(
    #     detect_landmark,
    #     frames,
    #     max_workers=num_workers,
    #     desc="Detecting Lip Movement"
    # )
    invalid_landmarks_ratio = sum(lnd is None for lnd in landmarks) / len(landmarks)
    logger.info(f"Current invalid frame ratio ({invalid_landmarks_ratio}) ")
    if invalid_landmarks_ratio > MAX_MISSING_FRAMES_RATIO:
        logging.info(
            "Invalid frame ratio exceeded maximum allowed ratio!! " +
            "Starting resizing the recorded video!!"
        )
        sequence = resize_frames(frames)
    else:
        # interpolate frames not being detected (if found).
        if invalid_landmarks_ratio != 0:
            logger.info("Linearly-interpolate invalid landmarks")
            continuous_landmarks = landmarks_interpolate(landmarks)
        else:
            continuous_landmarks = landmarks
        # crop mouth regions
        logger.info("Cropping the mouth region.")
        sequence = crop_patch(
            frames,
            len(frames),
            continuous_landmarks,
            MEAN_FACE_LANDMARKS,
        )
    # return lip-movement frames
    save_video(sequence, out_lip_filepath, fps=25)


def process_input_video(
    model_type: str,
    input_video_path: str,
    noise_snr: int,
    noise_type: str,
    request: gr.Request
):
    logger.info(20*'=' + " Starting a new request " + '='*20)
    logger.info("Request client: " + str(request.client))
    logger.info("Request headers: " + str(request.headers))
    if input_video_path is None:
        raise IOError(
            "Gradio didn't record the video. Refresh the web page, please!!"
        )
    
    # define (to be) output files
    outpath = OUTPATH / str(int(time.time()))
    outpath.mkdir(parents=True, exist_ok=True)
    audio_filepath = outpath / "audio.wav"
    video_filepath = outpath / "video.mp4"
    noisy_audio_filepath = outpath / "noisy_audio.wav"
    lip_video_filepath = outpath / "lip_movement.mp4"

    # start the lip movement preprocessing pipeline
    extract_lip_movement(
        input_video_path, video_filepath, lip_video_filepath,
        num_workers=min(os.cpu_count(), 5)
    )
    
    # mix audio with noise
    logger.info(f"Mixing audio with `{noise_type}` noise (SNR={noise_snr}).")
    noise_wav_files = NOISE[noise_type]
    noise_wav_file = noise_wav_files[random.randint(0, len(noise_wav_files)-1)]
    logger.debug(f"Noise Wav used is {noise_wav_file}")
    mixed = mix_audio_with_noise(
        input_video_path, audio_filepath, noisy_audio_filepath,
        noise_wav_file, noise_snr
    )
    
    # combine (audio+noise) with lip-movement
    logger.info("Adding noisy audio with the lip-movement video.")
    noisy_lip_filepath = outpath/"noisy_lip_movement.mp4"
    FFmpeg(
        inputs={noisy_audio_filepath: None, lip_video_filepath: None},
        outputs={noisy_lip_filepath: "-v quiet -c:v copy -c:a aac"},
    ).run()

    # Infer Audio-Video using Av-HuBERT
    av_text = infer_av_hubert(
        AV_RESOURCES[model_type]["model"],
        AV_RESOURCES[model_type]["task"],
        AV_RESOURCES[model_type]["generator"],
        lip_video_filepath,
        noisy_audio_filepath,
        duration=len(mixed)/16000
    )
    logger.info(f"Av-HuBERT Output: {av_text}")

    logger.info("Summary:")
    for k, v in TIME_TRACKER.items():
        logger.info(f'Function {k} executed in {v} seconds')
    logger.info(30*'=' + " Done! " + '='*30)
    return (str(noisy_lip_filepath), av_text)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avhubert-path", type=Path, required=True,
        help="Relative/Absolute path where avhubert repo is located."
    )
    parser.add_argument(
        "--av-models-path", type=Path, required=True,
        default=Path(__file__).parent / "av_models",
        help="Relative/Absolute path where AV checkpoints are located."
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        default=Path(__file__).parent / "recordings",
        help="Relative/Absolute path where output files will be written."
    )
    args = parser.parse_args()

    # start loading resources
    logger.info("Loading noise samples..")
    noise_path = args.output_path
    NOISE = load_noise_samples(noise_path)
    
    logger.info("Loading AV models!")
    if not args.av_models_path.exists():
        raise ValueError(
            f"av-models-path: `{args.av_models_path}` doesn't exist!!"
        )
    utils.import_user_module(
        argparse.Namespace(user_dir=str(args.avhubert_path))
    )
    AV_RESOURCES = load_av_models(args.av_models_path)
    
    logger.info("Loading models responsible for preprocessing!")
    metadata_path = args.output_path / "metadata"
    DETECTOR, PREDICTOR, MEAN_FACE_LANDMARKS = (
        load_needed_models_for_lip_movement(metadata_path)
    )
    logger.info("Done loading!")

    # set output directory
    OUTPATH = args.output_path
    OUTPATH.mkdir(parents=True, exist_ok=True)

    # cache already recorded videos
    VIDEOS_CACHE = {}
    logger.info("Caching previously recorded videos!")
    for hash_path in OUTPATH.rglob("*.md5"):
        with open(hash_path) as fin:
            md5hash = fin.read()
            VIDEOS_CACHE[md5hash] = hash_path.parent
    
    # define input interfaces
    logger.info("Building Gradio Interface!")
    webcam_interface = gr.Interface(
        fn=process_input_video,
        inputs=[
            gr.components.Dropdown(
                label="Model", choices=list(AV_RESOURCES.keys()),
                value=sorted(AV_RESOURCES.keys())[0]
            ),
            gr.components.Video(
                label="Webcam", source="webcam", include_audio=True,
                mirror_webcam=False
            ),
            gr.components.Slider(
                label="SNR", minimum=-20, maximum=20, step=5, value=20
            ),
            gr.components.Radio(
                label="Noise", choices=list(NOISE.keys()),
                value=sorted(NOISE.keys())[0]
            ),
        ],
        outputs=[
            gr.components.Video(
                label="Lip Movement", source="upload", include_audio=True
            ).style(height=256, width=256),
            gr.components.Text(label="AV-HuBERT Output"),
        ],
        allow_flagging="never",
    )
    OPTIONS = [ lang+"_avsr" for lang in ["ar", "en", "de", "el", "es", "fr", "it", "pt", "ru"]]
    OPTIONS += [ f"en-{lang}_avst" for lang in ["el", "es", "fr", "it", "pt", "ru"]]
    OPTIONS += [ lang+"-en_avst" for lang in ["el", "es", "fr", "it", "pt", "ru"]]
    upload_interface = gr.Interface(
        fn=process_input_video,
        inputs=[
            gr.components.Dropdown(
                label="Model", choices=OPTIONS,
                value="en_avsr"
            ),
            gr.components.Video(
                label="Video", source="upload", include_audio=True,
            ),
            gr.components.Slider(
                label="SNR", minimum=-20, maximum=20, step=5, value=20
            ),
            gr.components.Radio(
                label="Noise", choices=list(NOISE.keys()),
                value=sorted(NOISE.keys())[0]
            ),
        ],
        outputs=[
            gr.components.Video(
                label="Lip Movement", source="upload", include_audio=True
            ).style(height=256, width=256),
            gr.components.Text(label="AV-HuBERT Output"),
        ],
        examples = [
            [None, "muavic_internal/demo/examples/jeff3_low.mp4", None, None],
            [None, "muavic_internal/demo/examples/fr_example_subtitled.mp4", None, None],
        ],
        allow_flagging="never",
    )
    
    # create main interface
    iface = gr.TabbedInterface(
        interface_list=[webcam_interface, upload_interface],
        tab_names=["Record Video", "Upload Video"]
    )
    iface.queue().launch(debug=False, show_error=True)
