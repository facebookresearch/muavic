# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import re
import cv2
import sox
import wget
import yt_dlp
import ffmpeg
import pickle
import tarfile
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import transform
from collections import deque
from urllib.error import HTTPError


def is_empty(path):
    return any(path.iterdir()) == False


def read_txt_file(txt_filepath):
    with open(txt_filepath) as fin:
        return (line.strip() for line in fin.readlines())


def write_txt_file(lines, out_txt_filepath):
    with open(out_txt_filepath, "w") as fout:
        fout.writelines("\n".join([ln.strip() for ln in lines]))


def normalize_text(text):
    PUNCS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~؟؛,’‘×÷"
    # remove sound-effect description
    text = re.sub(r"\([^)]*\)", "", text)
    # remove punctuations
    text = text.translate(str.maketrans("", "", PUNCS))
    # normalize case
    text = text.lower()
    return text.strip()


def download_file(url, download_path):
    filename = url.rpartition("/")[-1]
    if not (download_path / filename).exists():
        try:
            # download file
            print(f"Downloading {filename} from {url}")
            custom_bar = (
                lambda current, total, width=80: wget.bar_adaptive(
                    round(current / 1024 / 1024, 2),
                    round(total / 1024 / 1024, 2),
                    width,
                )
                + " MB"
            )
            wget.download(url, out=str(download_path / filename), bar=custom_bar)
        except Exception as e:
            message = f"Downloading {filename} failed!"
            raise HTTPError(e.url, e.code, message, e.hdrs, e.fp)
    return True


def extract_tgz(tgz_filepath, extract_path, out_filename=None):
    if not tgz_filepath.exists():
        raise FileNotFoundError(f"{tgz_filepath} is not found!!")
    tgz_filename = tgz_filepath.name
    tgz_object = tarfile.open(tgz_filepath)
    if not out_filename:
        out_filename = tgz_object.getnames()[0]
    # check if file is already extracted
    if not (extract_path / out_filename).exists():
        for mem in tqdm(tgz_object.getmembers(), desc=f"Extracting {tgz_filename}"):
            out_filepath = extract_path / mem.get_info()["name"]
            if mem.isfile() and not out_filepath.exists():
                tgz_object.extract(mem, path=extract_path)
    tgz_object.close()


def download_extract_file_if_not(url, tgz_filepath, download_filename):
    download_path = tgz_filepath.parent
    if not tgz_filepath.exists():
        # download file
        download_file(url, download_path)
    # extract file
    extract_tgz(tgz_filepath, download_path, download_filename)


def load_meanface_metadata(metadata_path):
    mean_face_filepath = metadata_path / "20words_mean_face.npy"
    if not mean_face_filepath.exists():
        download_file(
            "https://dl.fbaipublicfiles.com/muavic/metadata/20words_mean_face.npy",
            metadata_path,
        )
    return np.load(mean_face_filepath)


def load_video_metadata(filepath):
    if not filepath.exists():
        # download & extract file
        lang_dir = filepath.parent.parent
        lang = lang_dir.name
        tgz_filepath = lang_dir.parent / f"{lang}_metadata.tgz"
        download_extract_file_if_not(
            url=f"https://dl.fbaipublicfiles.com/muavic/metadata/{lang}_metadata.tgz",
            tgz_filepath=tgz_filepath,
            download_filename=lang
        )
    if not filepath.exists():
        # file doesn't have metadata
        return None
    assert filepath.exists(), f"{filepath} should've been downloaded!"
    with open(filepath, "rb") as fin:
        metadata = pickle.load(fin)
    return metadata


def download_video_from_youtube(download_path, yt_id):
    """Downloads a video from YouTube given its id on YouTube"""
    video_out_path = download_path / f"{yt_id}.mp4"
    if video_out_path.exists():
        downloaded = True
    else:
        url = f"https://www.youtube.com/watch?v={yt_id}"
        # downloads the best `mp4` audio/video resolution.
        # TODO: download only video (no audio)
        ydl_opts = {"quiet": True, "format": "mp4", "outtmpl": str(video_out_path)}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
                downloaded = True
            except yt_dlp.utils.DownloadError:
                downloaded = False
    return downloaded


# def save_video(frames, out_filepath, fps):
#     height, width, _ = frames[0].shape
#     writer = cv2.VideoWriter(
#             filename=out_filepath,
#             fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
#             fps=float(fps),
#             frameSize=(width, height)
#         )
#     for frame in frames:
#         writer.write(frame)
#     writer.release()


def resize_frames(input_frames, new_size):
    resized_frames = []
    for frame in input_frames:
        try:
            resized_frames.append(cv2.resize(frame, new_size))
        except:
            pass #some frames are corrupt or missing
    return resized_frames


def get_audio_duration(audio_filepath):
    return sox.file_info.duration(audio_filepath)


def get_video_duration(video_filepath):
    try:
        streams = ffmpeg.probe(video_filepath)["streams"]
        for stream in streams:
            if stream["codec_type"] == "video":
                return float(stream["duration"])
    except:
        warnings.warn(f"Video file: `{video_filepath}` is corrupted... skipping!!")
        return -1


def get_video_resolution(video_filepath):
    for stream in ffmpeg.probe(video_filepath)["streams"]:
        if stream["codec_type"] == "video":
            height = int(stream["height"])
            width = int(stream["width"])
            return height, width
    raise TypeError(f"Input file: {video_filepath} doesn't have video stream!")


def get_audio_video_info(audio_path, video_path, fid):
    audio_filepath = audio_path / f"{fid}.wav"
    video_filepath = video_path / f"{fid}.mp4"
    audio_frames = (
        int(get_audio_duration(audio_filepath) * 16_000)
        if audio_filepath.exists()
        else -1
    )
    video_frames = (
        int(get_video_duration(video_filepath) * 25) if video_filepath.exists() else -1
    )
    return {
        "id": fid,
        "video": str(video_filepath),
        "audio": str(audio_filepath),
        "video_frames": video_frames,
        "audio_samples": audio_frames,
    }


def split_video_to_frames(video_filepath, fstart=None, fend=None, out_fps=25):
    # src: https://github.com/kylemcdonald/python-utils/blob/master/ffmpeg.py
    #NOTE: splitting video into frames is faster on CPU than GPU
    width, height = get_video_resolution(video_filepath)
    video_stream = ffmpeg.input(str(video_filepath)).video.filter("fps", fps=out_fps)
    channels = 3
    try:
        if fstart is not None and fend is not None:
            process = (
                video_stream.trim(start_frame=fstart, end_frame=fend)
                .setpts("PTS-STARTPTS")
                .output("pipe:", format="rawvideo", pix_fmt="bgr24")
                .run_async(pipe_stdout=True, quiet=True)
            )
            frames_counter = 0
            while frames_counter < fend - fstart:
                in_bytes = process.stdout.read(width * height * channels)
                in_frame = np.frombuffer(in_bytes, np.uint8).reshape(
                    width, height, channels
                )
                yield in_frame
                frames_counter += 1
        else:
            process = (
                video_stream.setpts("PTS-STARTPTS")
                .output("pipe:", format="rawvideo", pix_fmt="bgr24")
                .run_async(pipe_stdout=True, quiet=True)
            )
            while True:
                in_bytes = process.stdout.read(width * height * channels)
                if not in_bytes:
                    break
                in_frame = np.frombuffer(in_bytes, np.uint8).reshape(
                    width, height, channels
                )
                yield in_frame

    finally:
        process.stdout.close()
        process.wait()


def save_video(frames, out_filepath, fps, vcodec="libx264"):
    if len(frames) == 0:
        warnings.warn(
            f"Video segment `{out_filepath.stem}` has no metadata..." +
            " skipping!!" 
        )
        return
    height, width, _ = frames[0].shape
    process = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="bgr24", s="{}x{}".format(width, height)
        )
        .output(str(out_filepath), pix_fmt="bgr24", vcodec=vcodec, r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )
    for _, frame in enumerate(frames):
        try:
            process.stdin.write(frame.astype(np.uint8).tobytes())
        except:
            print(process.stderr.read())
    process.stdin.close()
    process.wait()


def load_video(filename):
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()


def warp_img(src, dst, img, std_size):
    tform = transform.estimate_transform(
        "similarity", src, dst
    )  # find the transformation matrix
    warped = transform.warp(
        img, inverse_map=tform.inverse, output_shape=std_size
    )  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype("uint8")
    return warped, tform


def apply_transform(trans, img, std_size):
    warped = transform.warp(img, inverse_map=trans.inverse, output_shape=std_size)
    warped = warped * 255  # note output from warp is double image (value range [0,1])
    warped = warped.astype("uint8")
    return warped


def cut_patch(img, metadata, height, width, threshold=5):
    center_x, center_y = np.mean(metadata, axis=0)
    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception("too much bias in height")
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception("too much bias in width")

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception("too much bias in height")
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception("too much bias in width")

    cutted_img = np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )
    return cutted_img


def crop_patch(
    video_frames,
    num_frames,
    metadata,
    mean_face_metadata,
    std_size=(256, 256),
    window_margin=12,
    start_idx=48,
    stop_idx=68,
    crop_height=96,
    crop_width=96,
):
    """Crop mouth patch"""
    stablePntsIDs = [33, 36, 39, 42, 45]
    margin = min(num_frames, window_margin)
    q_frame, q_metadata = deque(), deque()
    sequence = []
    for frame_idx, frame in enumerate(video_frames):
        if frame_idx >= len(metadata):
            break  #! Sadly, this is necessary
        q_metadata.append(metadata[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_metadata = np.mean(q_metadata, axis=0)
            cur_metadata = q_metadata.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img(
                smoothed_metadata[stablePntsIDs, :],
                mean_face_metadata[stablePntsIDs, :],
                cur_frame,
                std_size,
            )
            trans_metadata = trans(cur_metadata)
            # -- crop mouth patch
            sequence.append(
                cut_patch(
                    trans_frame,
                    trans_metadata[start_idx:stop_idx],
                    crop_height // 2,
                    crop_width // 2,
                )
            )

    while q_frame:
        cur_frame = q_frame.popleft()
        # -- transform frame
        trans_frame = apply_transform(trans, cur_frame, std_size)
        # -- transform metadata
        trans_metadata = trans(q_metadata.popleft())
        # -- crop mouth patch
        sequence.append(
            cut_patch(
                trans_frame,
                trans_metadata[start_idx:stop_idx],
                crop_height // 2,
                crop_width // 2,
            )
        )
    return sequence


def read_av_manifest(tsv_filepath):
    with open(tsv_filepath) as fin:
        res = []
        for ln in fin.readlines()[1:]:
            id_, video, audio, video_frames, audio_samples = ln.strip().split("\t")
            res.append(
                {
                    "id": id_,
                    "video": video,
                    "audio": audio,
                    "video_frames": video_frames,
                    "audio_samples": audio_samples,
                }
            )
    df = pd.DataFrame(res)
    df["video_frames"] = df["video_frames"].astype(int)
    df["audio_samples"] = df["audio_samples"].astype(int)
    return df


def write_av_manifest(df, out_filepath):
    with open(out_filepath, "w") as fout:
        fout.write("/\n")
    df.to_csv(out_filepath, sep="\t", header=False, index=False, mode="a")
