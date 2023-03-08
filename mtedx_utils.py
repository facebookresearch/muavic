import os
import cv2
import sox
import shutil
import ffmpeg
import tempfile
import pandas as pd
from functools import partial
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

from utils import *




def download_mtedx_data(download_path, src, tgt):
    """Downloads mTEDx data from OpenSLR"""
    tgz_filename = f"mtedx_{src}-{tgt}.tgz" if src != tgt else f"mtedx_{src}.tgz"
    download_extract_file_if_not(
        url = f"https://www.openslr.org/resources/100/{tgz_filename}",
        tgz_filepath = download_path / tgz_filename
    )


def download_mtedx_lang_videos(mtedx_path, src_lang):
    # keep track of non-found videos on YouTube
    try:
       not_found_videos = set(read_txt_file(mtedx_path / "not_found_videos.txt"))
    except FileNotFoundError:
        not_found_videos = set()
    # get files id per split
    for split in ["train", "valid", "test"]:
        out_path = mtedx_path / "video" / src_lang / split
        out_path.mkdir(parents=True, exist_ok=True)
        if not is_empty(out_path):
            if split == "train":
                print(f"\nDownloading {src_lang} videos from YouTube")
            # get youtube-ids from audio filenames inside `wav` directory
            wav_dir_path = mtedx_path/f"{src_lang}-{src_lang}"/"data"/split/"wav"
            yt_ids = [wav_filepath.stem for wav_filepath in wav_dir_path.glob('*')]
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
    with open(mtedx_path / "not_found_videos.txt", 'w') as fout:
        fout.writelines([f"{id_}\n" for id_ in not_found_videos])


def segment_normalize_audio_file(
    out_dir, in_file_info, out_sr=16_000, out_channels=1, out_format="wav"
):
    in_filepath, fid, seg_id, start_sec, end_sec = in_file_info
    out_filepath = out_dir / fid / f"{seg_id}.{out_format}"
    if not in_filepath.exists() or out_filepath.exists():
        return
    out_filepath.parent.mkdir(parents=True, exist_ok=True)
    tfm = sox.Transformer()
    tfm.set_output_format(rate=out_sr, channels=out_channels)
    tfm.trim(start_sec, end_sec)
    tfm.build_file(
        input_filepath=str(in_filepath),
        output_filepath=str(out_filepath)
    )


def preprocess_mtedx_audio(mtedx_path, src_lang, muavic_path):
    # get files id per split
    for split in ["train", "valid", "test"]:
        # create directory for segmented & normalized audio
        out_path = muavic_path / src_lang / "audio" / split
        out_path.mkdir(parents=True, exist_ok=True)
        if not is_empty(out_path):
            if split == "train":
                print(f"\nSegmenting {src_lang} audio files")
            # collect needed info from segment file
            segments_info = []
            split_dir_path = mtedx_path / f"{src_lang}-{src_lang}" / "data" / split
            wav_dir_path = split_dir_path / "wav"
            segment_file = split_dir_path / "txt" / "segments"
            
            for line in read_txt_file(segment_file):
                seg_id, fid, start, end = line.strip().split(' ')
                segments_info.append(
                    (wav_dir_path/(fid+".flac"), fid, seg_id, float(start), float(end))
                )
            # preprocess audio files
            process_map(
                partial(segment_normalize_audio_file, out_path),
                segments_info,
                max_workers=os.cpu_count(),
                desc=f"Preprocessing {src_lang}/{split} Audios",
                chunksize=1,
            )


def segment_normalize_video_file(
    mean_face_metadata, metadata_path, in_path, out_path, video_info
):
    out_fps = 25
    video_format = "mp4"
    video_id, video_segments = video_info
    in_filepath = in_path / f"{video_id}.{video_format}"
    video_metadata = load_video_metadata(metadata_path / f"{video_id}.pkl")
    tmp_dir = tempfile.mkdtemp()
    (
        ffmpeg
        .input(str(in_filepath)).video
        .filter('fps', fps=25)
        .output(f"{tmp_dir}/%d.png", start_number=0)
        .run(quiet=True)
    )
    for seg in video_segments:
        out_filepath = out_path / video_id / f"{seg['id']}.{video_format}"
        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        # start segmenting
        seg_metadata_length = len(video_metadata[seg["id"]])
        fstart = round(seg["start_sec"] * out_fps)
        fend = (
            round(seg["end_sec"] * out_fps)
            if seg_metadata_length == 0
            else
            fstart+seg_metadata_length
        ) # handles minor mismatches between metadata-length and video length
        if fstart == fend: fend = fstart+1 #adding 1-frame
        num_frames = fend-fstart
        video_frames = (
            cv2.imread(f"{tmp_dir}/{i}.png")
            for i in range(fstart, fend)
        )
        if len(video_metadata[seg["id"]]) > 0:
            frames = crop_patch(
                    video_frames,
                    num_frames,
                    video_metadata[seg["id"]],
                    mean_face_metadata,
                    std_size=(256,256),
                )
        else:
            frames = [
                cv2.resize(frame, (96, 96))
                for frame in video_frames
            ]
        # save video
        save_video(frames, out_filepath, out_fps)
    shutil.rmtree(tmp_dir)


def preprocess_mtedx_video(mtedx_path, metadata_path, src_lang, muavic_path):
    # load all metadata for all video files in the given language
    mean_face_filepath = metadata_path / "20words_mean_face.npy"
    if not mean_face_filepath.exists():
        download_file(
            "https://dl.fbaipublicfiles.com/muavic/metadata/20words_mean_face.npy",
            metadata_path
        )
    mean_face_metadata = np.load(mean_face_filepath)
    # get files id per split
    for split in ["train", "valid", "test"]:
        out_path = muavic_path / src_lang / "video" / split
        out_path.mkdir(parents=True, exist_ok=True)
        if not is_empty(out_path):
            if split == "train":
                print(
                    f"\nSegmenting {src_lang} videos files" + 
                    "(It takes a few hours to complete)")
        segment_file = (
            mtedx_path / f"{src_lang}-{src_lang}" / "data" / split / "txt" / "segments"
        )
        
        # collect needed info from segment file
        video_segments = defaultdict(list)
        for line in read_txt_file(segment_file):
            seg_id, fid, start_sec, end_sec = line.strip().split()
            video_segments[fid].append({
                "id": seg_id,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            })
        
        video_dir_path = mtedx_path / "video" / src_lang / split
        process_map(
            partial(
                segment_normalize_video_file,
                mean_face_metadata,
                metadata_path / src_lang / split,
                video_dir_path,
                out_path,
            ),
            video_segments.items(),
            max_workers=os.cpu_count(),
            chunksize=1,
        )


def get_mtedx_fileids(mtedx_path, lang_pair, split):
    # get segments file in mtedx dataset
    segments_path = (
        mtedx_path / lang_pair / "data" / split / "txt" / "segments"
    )
    # TODO: filter ids based on existing audio & video
    return (
        f"{split}/{ln.split()[0].rpartition('_')[0]}/{ln.split()[0]}"
        for ln in read_txt_file(segments_path)
    )


def get_audio_video_info(audio_path, fileid):
    audio_filepath = audio_path / f"{fileid}.wav"
    video_filepath = (
        str(audio_filepath)
        .replace("audio", "video")
        .replace("wav", "mp4")
    )
    #TODO: handle the case if either audio or video or both are not found!
    dur = sox.file_info.duration(audio_filepath)
    return {
        "id": fileid,
        "video": video_filepath,
        "audio": audio_filepath,
        "video_frames": round(dur*25),
        "audio_samples": round(dur*16_000),
        }


def prepare_mtedx_manifest(mtedx_path, src_lang, muavic_path):
    for split in ["train", "valid", "test"]:
        manifest_filepath = muavic_path / src_lang / f"{split}.tsv"
        if not manifest_filepath.exists():
            if split == "train":
                print(f"\nCreating manifests for {src_lang}")
            lang_pair = f"{src_lang}-{src_lang}"
            fileids = list(get_mtedx_fileids(mtedx_path, lang_pair, split))
            audio_datapath = muavic_path / src_lang / "audio"
            av_manifest_df = pd.DataFrame(
                process_map(
                    partial(get_audio_video_info, audio_datapath),
                    fileids,
                    desc=f"Creating {src_lang}/{split} manifest",
                    max_workers=os.cpu_count(),
                    chunksize=1
                )
            )
            # write down the manifest TSV file
            with open(manifest_filepath, 'w') as fout: fout.write('/\n')
            av_manifest_df.to_csv(
                manifest_filepath, sep='\t', header=False, index=False, 
                mode='a'
            )


def prepare_mtedx_translation(mtedx_path, mt_trans_path, lang, muavic_path):
    # download & extradt pseudo-translation if that wasn't done already
    lang_pair = f"{lang}-en"
    tgz_filename = f"{lang_pair}.tgz"
    tgz_filepath = mt_trans_path / tgz_filename
    url = f"https://dl.fbaipublicfiles.com/muavic/mt_trans/{tgz_filename}",
    download_extract_file_if_not(url, tgz_filepath)
    
    for split in ["train", "valid", "test"]:
        # set output files
        out_src_filepath = muavic_path / lang / f"{split}.{lang}"
        out_tgt_filepath = muavic_path / lang / f"{split}.en"
        if not out_src_filepath.exists() or not out_tgt_filepath.exists():
            if split == "train":
                print(f"\nCollecting translation data for {lang}-en")
            # combine human translation with mt translation for "train" & "valid"
            if split != "test":
                # load pseudo-translations from mt_trans_path
                pseudo_trans = pd.DataFrame({
                    "id": list(
                        get_mtedx_fileids(mtedx_path, f"{lang}-{lang}", split)
                    ),
                    lang: read_txt_file(
                        mtedx_path /  f"{lang}-{lang}" / "data" / split / "txt" / f"{split}.{lang}"
                    ),
                    "en": read_txt_file(
                        mt_trans_path /  f"{lang}-en" / f"{split}.en"
                    ),
                }).set_index("id")

                # load real translations from mtedx_path
                mtedx_trans_path = mtedx_path / f"{lang}-en" / "data" / split / "txt"
                human_trans = pd.DataFrame({
                    "id": list(
                        get_mtedx_fileids(mtedx_path, f"{lang}-en", split)
                    ),
                    lang: read_txt_file(
                        mtedx_trans_path / f"{split}.{lang}"
                    ),
                    "en": read_txt_file(
                        mtedx_trans_path / f"{split}.en"
                    ),
                }).set_index("id")

                # combine pseudo translation with human translation
                pseudo_trans.update(human_trans)
                all_trans = pseudo_trans.copy()

                write_txt_file(all_trans[lang].tolist(), out_src_filepath)
                write_txt_file(all_trans["en"].tolist(), out_tgt_filepath)
            
            # use only human translation for "test"
            else:
                mtedx_trans_path = mtedx_path / f"{lang}-en" / "data" / split / "txt"
                # copy source language
                shutil.copyfile(
                    src = mtedx_trans_path/f"test.{lang}",
                    dst = out_src_filepath
                )
                # copy target language
                shutil.copyfile(
                    src = mtedx_trans_path / "test.en",
                    dst = out_tgt_filepath
                )

