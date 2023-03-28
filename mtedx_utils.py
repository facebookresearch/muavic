# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import cv2
import sox
import shutil
import tempfile
import warnings
import pandas as pd
from functools import partial
from collections import defaultdict, OrderedDict
from tqdm.contrib.concurrent import process_map

from utils import *


# define global constants
SPLITS = ["train", "valid", "test"]


def download_mtedx_data(download_path, src, tgt):
    """Downloads mTEDx data from OpenSLR"""
    tgz_filename = f"mtedx_{src}-{tgt}.tgz" if src != tgt else f"mtedx_{src}.tgz"
    download_extract_file_if_not(
        url=f"https://www.openslr.org/resources/100/{tgz_filename}",
        tgz_filepath=download_path / tgz_filename,
        download_filename=f"{src}-{tgt}"
    )


def download_mtedx_lang_videos(mtedx_path, src_lang):
    # keep track of non-found videos on YouTube
    try:
        not_found_videos = set(read_txt_file(mtedx_path / "not_found_videos.txt"))
    except FileNotFoundError:
        not_found_videos = set()
    # get files id per split
    for split in SPLITS:
        out_path = mtedx_path / "video" / src_lang / split
        out_path.mkdir(parents=True, exist_ok=True)
        if is_empty(out_path): #TODO: better check
            if split == "train":
                print(f"\nDownloading {src_lang} videos from YouTube")
            # get youtube-ids from audio filenames inside `wav` directory
            wav_dir_path = (
                mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "wav"
            )
            yt_ids = [wav_filepath.stem for wav_filepath in wav_dir_path.glob("*")]
            # download videos from YouTube
            downloading_status = process_map(
                partial(download_video_from_youtube, out_path),
                yt_ids,
                max_workers=os.cpu_count(),
                desc=f"Downloading {src_lang}/{split} Videos",
                chunksize=1,
            )
            assert len(yt_ids) == len(downloading_status)
            for yt_id, downloaded in zip(yt_ids, downloading_status):
                if not downloaded:
                    not_found_videos.add(yt_id)
    with open(mtedx_path / "not_found_videos.txt", "w") as fout:
        fout.writelines([f"{id_}\n" for id_ in not_found_videos])


def segment_normalize_audio_file(out_dir, in_file_info):
    out_sr = 16_000
    out_channels = 1
    out_format = "wav"
    in_filepath, fid, seg_id, start_sec, end_sec = in_file_info
    out_filepath = out_dir / fid / f"{seg_id}.{out_format}"
    if not in_filepath.exists() or out_filepath.exists():
        return
    out_filepath.parent.mkdir(parents=True, exist_ok=True)
    tfm = sox.Transformer()
    tfm.set_output_format(rate=out_sr, channels=out_channels)
    tfm.trim(start_sec, end_sec)
    tfm.build_file(str(in_filepath), str(out_filepath))


def preprocess_mtedx_audio(mtedx_path, src_lang, muavic_path):
    for split in SPLITS:
        split_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split
        audio_segments = list(read_txt_file(split_dir_path / "txt" / "segments"))
        # create directory for segmented & normalized audio
        out_path = muavic_path / src_lang / "audio" / split
        out_path.mkdir(parents=True, exist_ok=True)
        # early-quit condition
        num_curr_audio_segments = len(list(out_path.rglob("*.wav")))
        if num_curr_audio_segments == len(audio_segments):
            continue # skip if all audio segments are already processed
        if split == "train":
            print(f"\nSegmenting {src_lang} audio files")
        # collect needed info from segment file
        segments_info = []
        wav_dir_path = split_dir_path / "wav"
        for line in audio_segments:
            seg_id, fid, start, end = line.strip().split(" ")
            segments_info.append(
                (
                    wav_dir_path / (fid + ".flac"),
                    fid,
                    seg_id,
                    float(start),
                    float(end),
                )
            )
        # preprocess audio files
        process_map(
            partial(segment_normalize_audio_file, out_path),
            segments_info,
            max_workers=os.cpu_count(),
            desc=f"Preprocessing {src_lang}/{split} Audios",
            chunksize=1,
        )


def segment_normalize_video(mean_face_metadata, in_path, out_path, seg_info):
    fps = 25
    out_format = "mp4"
    out_filepath = out_path / f"{seg_info['id']}.{out_format}"
    # skip if file is already segmented
    if out_filepath.exists():
        return
    # start segmenting
    seg_metadata_length = len(seg_info["metadata"])
    fstart = round(seg_info["start"] * fps)
    fend = (
        round(seg_info["end"] * fps)
        if seg_metadata_length == 0
        else fstart + seg_metadata_length
    )  # handles minor mismatches between metadata-length and video-length
    if fstart == fend:
        fend = fstart + 1  # adding 1-frame
    num_frames = fend - fstart
    vid_frames = (
        cv2.imread(f"{in_path}/{i}.png")
        for i in range(fstart, fend)
        if os.path.exists(f"{in_path}/{i}.png")
    )
    if seg_metadata_length > 0:
        # detect the mouth ROI and crop it
        frames = crop_patch(
            vid_frames,
            num_frames,
            seg_info["metadata"],
            mean_face_metadata,
        )
    else:
        # resize the frames (since there were no metadata for it)
        frames = resize_frames(vid_frames, new_size=(96, 96))
    # save video
    save_video(frames, out_filepath, fps)


def preprocess_mtedx_video(mtedx_path, metadata_path, src_lang, muavic_path):
    mean_face_metadata = load_meanface_metadata(metadata_path)
    for split in SPLITS:
        split_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split
        video_segments = list(read_txt_file(split_dir_path / "txt" / "segments"))
        # create directory for segmented & normalized video
        out_path = muavic_path / src_lang / "video" / split
        out_path.mkdir(parents=True, exist_ok=True)
        # early-quit condition (might be TOO STRICT)
        num_curr_video_segments = len(list(out_path.rglob("*.mp4")))
        if num_curr_video_segments == len(video_segments):
            continue # skip if all video segments are already processed
        if split == "train":
            print(
                f"\nSegmenting `{src_lang}` videos files "
                + "(It takes a few hours to complete)"
            )
        # collect needed info from segment file
        segment_file = (
            mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "segments"
        )
        video_to_segments = defaultdict(list)
        for line in read_txt_file(segment_file):
            seg_id, fid, start_sec, end_sec = line.strip().split()
            video_to_segments[fid].append(
                {
                    "id": seg_id,
                    "start": float(start_sec),
                    "end": float(end_sec),
                }
            )
        # sort videos by the number of segments in ascending order
        video_to_segments = OrderedDict(
            sorted(video_to_segments.items(), key=lambda x: len(x[1]))
        )
        out_fps = 25
        video_format = "mp4"
        in_video_dir_path = mtedx_path / "video" / src_lang / split
        for video_id, video_segments in tqdm(video_to_segments.items()):
            # check if video has been already processed:
            all_segments_are_processed = all(
                (out_path / video_id / f"{seg['id']}.{video_format}").exists()
                for seg in video_segments
            )
            if all_segments_are_processed:
                continue
            # prepare to process video file
            in_filepath = in_video_dir_path / f"{video_id}.{video_format}"
            if not in_filepath.exists():
                warnings.warn(
                    f"TED talk `{in_filepath.stem}` hasn't been downloaded..." +
                    " skipping!!" 
                )
                continue
            tmp_dir_path = tempfile.mkdtemp()
            (
                ffmpeg.input(str(in_filepath))
                .video.filter("fps", fps=out_fps)
                .output(f"{tmp_dir_path}/%d.png", start_number=0)
                .run(quiet=True)
            )
            # load metadata
            video_metadata = load_video_metadata(
                metadata_path / src_lang / split / f"{video_id}.pkl"
            )
            if video_metadata is None:
                warnings.warn(
                    f"TED talk `{in_filepath.stem}` doesn't have metadata..." +
                    " skipping!!"
                )
                continue
            # add metadata to video_segments
            for seg in video_segments:
                seg["metadata"] = video_metadata.get(seg["id"], [])
            # set the output path for the video segments
            out_seg_path = out_path / video_id
            out_seg_path.mkdir(parents=True, exist_ok=True)
            # start segmenting video into smaller segments and process them
            process_map(
                partial(
                    segment_normalize_video,
                    mean_face_metadata,
                    tmp_dir_path,
                    out_seg_path,
                ),
                video_segments,
                max_workers=min(len(video_segments), os.cpu_count()),
                chunksize=1,
                disable=True,  # disables progress bar
            )
            shutil.rmtree(tmp_dir_path)


def get_mtedx_fileids(segment_filepath):
    # get segments file in mtedx dataset
    return [ln.split()[0] for ln in read_txt_file(segment_filepath)]


def prepare_mtedx_avsr_manifests(mtedx_path, lang, muavic_path):
    for split in SPLITS:
        out_manifest_filepath = muavic_path / lang / f"{split}_avsr.tsv"
        if not out_manifest_filepath.exists():
            if split == "train":
                print(f"\nCreating AVSR manifests for `{lang}`")
            mtedx_txt_dir_path = mtedx_path / f"{lang}-{lang}" / "data" / split / "txt"
            audio_datapath = muavic_path / lang / "audio" / split
            video_datapath = muavic_path / lang / "video" / split
            fileids = get_mtedx_fileids(mtedx_txt_dir_path / "segments")
            # get audio/video frames
            av_manifest_df = pd.DataFrame(
                process_map(
                    partial(get_audio_video_info, audio_datapath, video_datapath),
                    fileids,
                    desc=f"Creating {lang}/{split} manifest",
                    max_workers=os.cpu_count(),
                    chunksize=1,
                )
            )
            # write down the manifest TSV file
            write_av_manifest(av_manifest_df, out_manifest_filepath)
            # copy language transcription
            shutil.copyfile(
                mtedx_txt_dir_path / f"{split}.{lang}",
                muavic_path / lang / f"{split}_avsr.{lang}",
            )


def prepare_mtedx_avst_manifests(mtedx_path, mt_trans_path, lang, muavic_path):
    # download & extract pseudo-translation if that wasn't done already
    lang_pair = f"{lang}-en"
    tgz_filename = f"{lang_pair}.tgz"
    tgz_filepath = mt_trans_path / tgz_filename
    url = f"https://dl.fbaipublicfiles.com/muavic/mt_trans/{tgz_filename}"
    download_extract_file_if_not(url, tgz_filepath, lang_pair)

    for split in tqdm(SPLITS):
        # set output files
        out_tgt_filepath = muavic_path / lang / "en" / f"{split}_avst.en"
        out_manifest_filepath = muavic_path / lang / "en" / f"{split}_avst.tsv"
        out_tgt_filepath.parent.mkdir(parents=True, exist_ok=True)
        if not out_tgt_filepath.exists():
            if split == "train":
                print(f"\nCreating AVST manifests for `{lang}-en`")
            # combine human translation with MT translation for "train" & "valid"
            if split != "test":
                # load pseudo-translations from mt_trans_path
                pseudo_trans = pd.DataFrame(
                    {
                        "id": read_txt_file(
                            mt_trans_path / f"{lang}-en" / f"{split}_id.txt"
                        ),
                        lang: read_txt_file(
                            mt_trans_path / f"{lang}-en" / f"{split}.{lang}"
                        ),
                        "en": read_txt_file(
                            mt_trans_path / f"{lang}-en" / f"{split}.en"
                        ),
                    }
                ).set_index("id")

                # load real translations from mtedx_path
                mtedx_trans_path = mtedx_path / f"{lang}-en" / "data" / split / "txt"
                human_trans = pd.DataFrame(
                    {
                        "id": get_mtedx_fileids(mtedx_trans_path / "segments"),
                        lang: read_txt_file(mtedx_trans_path / f"{split}.{lang}"),
                        "en": read_txt_file(mtedx_trans_path / f"{split}.en"),
                    }
                ).set_index("id")

                # combine pseudo translation with human translation
                pseudo_trans.update(human_trans)
                write_txt_file(pseudo_trans["en"].tolist(), out_tgt_filepath)
                # copy AVSR manifest to be AVST's
                shutil.copyfile(
                    src=muavic_path / lang / f"{split}_avsr.tsv",
                    dst=out_manifest_filepath,
                )

            # use only human translation for "test"
            else:
                mtedx_trans_path = mtedx_path / f"{lang}-en" / "data" / split / "txt"
                # copy target language
                shutil.copyfile(src=mtedx_trans_path / f"test.en", dst=out_tgt_filepath)
                # copy AVSR manifest to be AVST's
                shutil.copyfile(
                    src=muavic_path / lang / f"{split}_avsr.tsv",
                    dst=out_manifest_filepath,
                )
