# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os, re
import shutil
import xmltodict
import itertools
import pandas as pd
from gzip import GzipFile
from functools import partial
from tqdm.contrib.concurrent import process_map

from utils import *


# define global constants
VIDEO_END = 24 * 3600 # big number to represent video end frame
SPLITS = ["train", "valid", "test"]
TARGET_LANGS = ["el", "es", "fr", "it", "pt", "ru"]


def create_manifest_for_pretrain(pretrain_path):
    def get_word_intervals(lines):
        word_intervals = []
        for i_line, ln in enumerate(lines):
            if ln.startswith("WORD"):
                start_index = i_line
                break
        for ln in lines[start_index + 1 :]:
            word, start, end, _ = ln.strip().split()
            word_intervals.append([word, float(start), float(end)])
        return word_intervals

    def combine_word_intervals(word_intervals, silence_duration):
        combined_word_intervals = []
        curr_sent = [word_intervals[0]]
        for word, start, end in word_intervals[1:]:
            assert (
                start >= curr_sent[-1][-1]
            ), f"{fid} , {word}, start-{start}, prev-{curr_sent[-1][-1]}"
            # check if silence between two words is longer than threshold
            if start - curr_sent[-1][-1] > silence_duration:
                # split & create a new sentence
                combined_word_intervals.append(curr_sent)
                curr_sent = [[word, start, end]]
            else:
                curr_sent.append([word, start, end])
        if len(curr_sent) > 0:
            combined_word_intervals.append(curr_sent)
        return combined_word_intervals

    # define needed variables
    max_duration = 15
    silence_threshold = 0.4
    df = {"fid": [], "sent": [], "start": [], "end": []}
    print("\nPrepare manifest to segment LRS3 `pretrain` set")
    txt_files = list(pretrain_path.rglob("*.txt"))
    for txt_filepath in tqdm(txt_files, desc="Creating pretrain manifest"):
        fid = str(txt_filepath.relative_to(pretrain_path))[:-4]
        lines = list(read_txt_file(txt_filepath))
        word_intervals = get_word_intervals(lines)
        # check if sentence is short
        last_word_end_sec = word_intervals[-1][-1]
        if last_word_end_sec < max_duration:
            # add it to our dataframe as it is
            raw_text = lines[0].strip().split(":")[-1].strip()
            df["fid"].append(fid)
            df["sent"].append(raw_text)
            df["start"].append(0)
            df["end"].append(VIDEO_END)
        else:
            # divide sentence into smaller segments using silence duration
            combined_word_intervals = combine_word_intervals(
                word_intervals, silence_threshold
            )
            for i, word_intervals in enumerate(combined_word_intervals):
                curr_start_sec = word_intervals[0][1]
                curr_end_sec = word_intervals[-1][2]
                # set start sec for the sentence
                if i == 0:
                    start_sec = 0
                else:
                    prev_end_sec = combined_word_intervals[i - 1][-1][2]
                    start_sec = (curr_start_sec + prev_end_sec) / 2
                # set the end sec for the sentence
                if i == len(combined_word_intervals) - 1:
                    end_sec = VIDEO_END
                else:
                    next_start_sec = combined_word_intervals[i + 1][0][1]
                    end_sec = (curr_end_sec + next_start_sec) / 2
                df["fid"].append(fid + "_" + str(i))
                df["sent"].append(" ".join([x[0] for x in word_intervals]))
                df["start"].append(round((start_sec), 3))
                df["end"].append(round(end_sec, 3))
    return pd.DataFrame(df)


def segment_video_and_text(in_path, out_path, info):
    # set input and output files
    vid_filename = info["fid"].split("_")[0]
    in_video_filepath = in_path / f"{vid_filename}.mp4"
    out_video_filepath = out_path / f"{info['fid']}.mp4"
    out_text_filepath = out_path / f"{info['fid']}.txt"
    out_text_filepath.parent.mkdir(parents=True, exist_ok=True)
    # write out text (if not found)
    if not out_text_filepath.exists():
        with open(out_text_filepath, "w") as fout:
            fout.write(f"Text:  {info['sent']}\n")
    # process video (if not already)
    if not out_video_filepath.exists():
        # check if video should be segmented
        if info["end"] == VIDEO_END:
            # video is short, copy it instead
            shutil.copyfile(in_video_filepath, out_video_filepath)
        else:
            # segment video
            in_video = ffmpeg.input(str(in_video_filepath))
            trimmed_video = (
                in_video
                .trim(start=info["start"], end=info["end"])
                .setpts('PTS-STARTPTS')
            )
            trimmed_audio = (
                in_video
                .filter_("atrim", start=info["start"], end=info["end"])
                .filter_("asetpts", "PTS-STARTPTS")
            )
            out_video = ffmpeg.concat(trimmed_video, trimmed_audio, v=1, a=1).node
            output = ffmpeg.output(
                out_video[0],
                out_video[1],
                str(out_video_filepath),
                vcodec="libx264",
                loglevel="quiet",
            )
            output.run(overwrite_output=True)


def segment_pretrain_videos_and_text(lrs3_path):
    seg_pretrain_path = lrs3_path / "seg_pretrain"
    seg_pretrain_path.mkdir(parents=True, exist_ok=True)
    if len(list(seg_pretrain_path.rglob("*.mp4"))) == 268864:
        return

    pretrain_path = lrs3_path / "pretrain"
    df = create_manifest_for_pretrain(pretrain_path)
    print("\nStart segmenting LRS3 `pretrain` set to `seg_pretrain`")
    process_map(
        partial(segment_video_and_text, pretrain_path, seg_pretrain_path),
        df.to_dict("records"),
        max_workers=os.cpu_count(),
        desc="Segmenting LRs3-pretrain",
        chunksize=1,
    )


def load_lrs3_valid_ids(metadata_path):
    filepath = metadata_path / "lrs3_valid_ids.txt"
    if not filepath.exists():
        download_file(
            "https://dl.fbaipublicfiles.com/muavic/metadata/lrs3_valid_ids.txt",
            metadata_path,
        )
    return set([ln.partition("/")[-1] for ln in read_txt_file(filepath)])


def extract_audio_video_from_video(
    mean_face_metadata, en_metadata_path, in_path, out_path, info
):
    in_split, out_split, fid = info
    # extract audio from video
    in_video_filepath = in_path / in_split / f"{fid}.mp4"
    out_audio_filepath = out_path / "audio" / out_split / f"{fid}.wav"
    out_audio_filepath.parent.mkdir(parents=True, exist_ok=True)
    if not out_audio_filepath.exists():
        (
            ffmpeg.input(str(in_video_filepath))
            .output(
                str(out_audio_filepath),
                acodec="pcm_s16le",
                ac=1,
                ar="16000",
                loglevel="quiet",
            )
            .run(overwrite_output=True)
        )
    # preprocess video
    out_video_filepath = out_path / "video" / out_split / f"{fid}.mp4"
    out_video_filepath.parent.mkdir(parents=True, exist_ok=True)
    if not out_video_filepath.exists():
        out_fps = 25
        # load landmark
        metadata_filename, _, seg_id = fid.rpartition("/")
        metadata_filepath = en_metadata_path / out_split / f"{metadata_filename}.pkl"
        seg_metadata = load_video_metadata(metadata_filepath)[seg_id]
        video_frames = split_video_to_frames(in_video_filepath)
        # load input video
        num_frames = min(
            len(seg_metadata), round(get_video_duration(in_video_filepath) * out_fps)
        )
        if len(seg_metadata) > 0:
            frames = crop_patch(
                video_frames,
                num_frames,
                seg_metadata,
                mean_face_metadata,
                std_size=(256, 256),
            )
        else:
            frames = resize_frames(video_frames, new_size=(96, 96))
        # save video
        save_video(frames, out_video_filepath, out_fps)


def process_lrs3_videos(lrs3_path, metadata_path, muavic_path):
    mean_face_metadata = load_meanface_metadata(metadata_path)
    for split in ["seg_pretrain", "trainval", "test"]:
        fids = [
            str(filepaths.relative_to(lrs3_path / split))[:-4]  # removes file ext
            for filepaths in (lrs3_path / split).rglob("*.mp4")
        ]
        # map LRS3 splits to either train, valid, or test.
        if split == "seg_pretrain":
            fids = [(split, "train", id_) for id_ in fids]
        elif split == "trainval":
            # divide "trainval" split into "train" & "valid"
            valid_fids = load_lrs3_valid_ids(metadata_path)
            fids = [
                (split, "train", id_)
                if id_ not in valid_fids
                else (split, "valid", id_)
                for id_ in fids
            ]
        elif split == "test":
            fids = [(split, "test", id_) for id_ in fids]
        # start processing data
        process_map(
            partial(
                extract_audio_video_from_video,
                mean_face_metadata,
                metadata_path / "en",
                lrs3_path,
                muavic_path / "en",
            ),
            fids,
            max_workers=os.cpu_count(),
            desc=f"Processing LRS3/{split}",
            chunksize=1,
        )


def prepare_lrs3_avsr_manifests(lrs3_path, muavic_path):
    # gather LRS3 textual data if transcription files haven't been written
    if len(list((muavic_path / "en").glob("*.en"))) != 3:
        fid_to_text = {}
        for split in ["seg_pretrain", "trainval", "test"]:
            if split == "seg_pretrain":
                print("\nGathering LRS3 textual data:")
            text_files = list((lrs3_path / split).rglob("*.txt"))
            for text_filepath in tqdm(text_files, f"collecting {split} text"):
                fid = str(text_filepath.relative_to(lrs3_path / split))[:-4]
                first_line = list(read_txt_file(text_filepath))[0]
                text = first_line.split(":")[-1].strip().lower()
                fid_to_text[fid] = text

    for split in SPLITS:
        # create transcription manifest
        transcription_filepath = muavic_path / "en" / f"{split}_avsr.en"
        if not transcription_filepath.exists():
            if split == "train":
                print(f"\nCreating AVSR manifests for `en`")
            # get split-specific file ids
            fids = [
                str(filepath.relative_to(muavic_path / "en" / "audio" / split))[:-4]
                for filepath in (muavic_path / "en" / "audio" / split).rglob("*.wav")
            ]
            transcriptions = [
                fid_to_text[fid]
                for fid in tqdm(fids, desc=f"en/{split} AVSR transcriptions")
            ]
            write_txt_file(transcriptions, transcription_filepath)
        # create AVSR manifest
        manifest_filepath = muavic_path / "en" / f"{split}_avsr.tsv"
        if not manifest_filepath.exists():
            audio_datapath = muavic_path / "en" / "audio" / split
            video_datapath = muavic_path / "en" / "video" / split
            av_manifest_df = pd.DataFrame(
                process_map(
                    partial(get_audio_video_info, audio_datapath, video_datapath),
                    fids,
                    desc=f"en/{split} AVSR manifest",
                    max_workers=os.cpu_count(),
                    chunksize=1,
                )
            )
            # write down the manifest TSV file
            write_av_manifest(av_manifest_df, manifest_filepath)


def extract_ted2020_data(tgz_filepath, src, tgt, out_path):
    """
    Parses a compressed tmx file and writes the data into a TSV file.
    """
    # parse compressed file
    tmx_dict = xmltodict.parse(GzipFile(tgz_filepath))
    # extract parallel sentences
    lang_pair = f"{src}-{tgt}"
    src_sents, tgt_sents = [], []
    for tu in tqdm(tmx_dict["tmx"]["body"]["tu"], desc=f"Extracting {lang_pair}"):
        src_sents.append(tu["tuv"][0]["seg"])
        tgt_sents.append(tu["tuv"][1]["seg"])
    # write files
    out_tsv = out_path / f"{lang_pair}.tsv"
    # NOTE: TED2020 orders languages alphabetically (src < tgt)
    if src > tgt:
        src, tgt = tgt, src
    pd.DataFrame(
        {
            src: src_sents,
            tgt: tgt_sents,
        }
    ).drop_duplicates().dropna().to_csv(out_tsv, sep="\t", index=False)


def download_ted2020(ted2020_path):
    # download TED2020 for target languages
    for lang in TARGET_LANGS:
        # download & extract TED2020 data if not
        if not (ted2020_path / f"en-{lang}.tsv").exists():
            tgz_filename = (
                f"{lang}-en.txt.zip" if lang < "en" else f"en-{lang}.txt.zip"
            )
            tgz_filepath = ted2020_path / tgz_filename
            if not tgz_filepath.exists():
                # download file
                download_file(
                    f"https://opus.nlpl.eu/download.php?f=TED2020/v1/moses/{tgz_filename}",
                    ted2020_path,
                )
            # extract file
            extract_ted2020_data(tgz_filepath, "en", lang, ted2020_path)


def segment_ted2020_sents(src_sents, tgt_sents):
    assert len(src_sents) == len(tgt_sents)
    # regex to represent end-of-sentence
    SENT_END_REGEX = re.compile(
        "(?<=(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Dr)(?<!Jr)[.!,?؟،])\s{1,2}"
    )  # src: https://ideone.com/25TUP2
    # create punctuation segmenter
    segmenter = lambda x: SENT_END_REGEX.split(x)
    # iterate over src/tgt sentences
    segmented_src_sents, segmented_tgt_sents = [], []
    for src_s, tgt_s in zip(src_sents, tgt_sents):
        # try to split sentences into smaller segments
        src_segments = segmenter(src_s)
        tgt_segments = segmenter(tgt_s)
        # only add them if we got same number of segments
        if len(src_segments) == len(tgt_segments):
            segmented_src_sents.extend(src_segments)
            segmented_tgt_sents.extend(tgt_segments)
        else:
            segmented_src_sents.append(src_s)
            segmented_tgt_sents.append(tgt_s)
    assert len(segmented_src_sents) == len(
        segmented_tgt_sents
    ), "TED2020 segmented translations don't match!!"
    return segmented_src_sents, segmented_tgt_sents


def prepare_lrs3_avst_manifests(mt_trans_path, ted2020_path, muavic_path):
    # start generating output translation files
    print(f"\nCreating AVST manifests")
    for lang in TARGET_LANGS:
        # load human translations from TED2020 dataset
        ted2020_df = pd.read_csv(ted2020_path / f"en-{lang}.tsv", sep="\t").dropna()
        # segment ted2020 sentences
        seg_en_sents, seg_lang_sents = segment_ted2020_sents(
            ted2020_df["en"].tolist(), ted2020_df[lang].tolist()
        )
        # use segmented translations (after normalization) as human translations
        human_trans = pd.DataFrame(
            {
                "en": [normalize_text(sent) for sent in seg_en_sents],
                lang: seg_lang_sents,
            }
        )
        for split in tqdm(SPLITS, desc=f"en-{lang} AVST manifest"):
            # set output files
            out_tgt_filepath = muavic_path / "en" / lang / f"{split}_avst.{lang}"
            out_manifest_filepath = muavic_path / "en" / lang / f"{split}_avst.tsv"
            out_tgt_filepath.parent.mkdir(parents=True, exist_ok=True)
            if not out_tgt_filepath.exists():
                # load AVSR manifest (AVST manifest is usually a subset)
                manifest_df = read_av_manifest(muavic_path / "en" / f"{split}_avsr.tsv")
                # combine human translation with MT translation for "train" & "valid"
                if split != "test":
                    # load pseudo-translation
                    split_ids = read_txt_file(
                        mt_trans_path / "en-x" / f"{split}_id.txt"
                    )
                    # remove split from id (e.g `test/x/xx_0`` -> `x/xx_0`)
                    split_ids = [id_.partition("/")[-1] for id_ in split_ids]
                    pseudo_trans = pd.DataFrame(
                        {
                            "id": split_ids,
                            "en": [
                                normalize_text(ln)
                                for ln in read_txt_file(
                                    mt_trans_path / "en-x" / f"{split}.en"
                                )
                            ],
                            lang: read_txt_file(
                                mt_trans_path / "en-x" / f"{split}.{lang}"
                            ),
                        }
                    ).set_index("id")
                    # add pseudo translation to manifest
                    pseudo_trans = pd.merge(
                        manifest_df, pseudo_trans, on="id", how="inner"
                    )
                    assert len(pseudo_trans) == len(
                        manifest_df
                    ), "Adding pseudo translation to AVSR manifest was wrong!!"

                    # merge using the English text
                    ### 1. get the matched indices
                    merged_trans = pd.merge(
                        left=pseudo_trans,
                        right=human_trans.drop_duplicates(subset=["en"]),
                        on="en",
                        how="left",
                    ).set_index(pseudo_trans.index)
                    assert len(merged_trans) == len(
                        pseudo_trans
                    ), "Mismatch between pseudo and human translation!!"
                    matched_indices = merged_trans[
                        merged_trans[f"{lang}_y"].notna()
                    ].index

                    ### 2. replace these indices with true translations
                    pseudo_trans.loc[matched_indices][lang] = merged_trans.loc[
                        matched_indices
                    ][f"{lang}_y"].values
                    assert len(pseudo_trans.dropna()) == len(
                        pseudo_trans
                    ), "Merging true translations with ids caused N/A values!!"

                    ### 3. write out the results
                    write_txt_file(pseudo_trans[lang].tolist(), out_tgt_filepath)
                    pseudo_trans.drop(["en", lang], axis=1, inplace=True)
                    write_av_manifest(pseudo_trans, out_manifest_filepath)

                # use only human translation for "test"
                else:
                    # add English transcriptions
                    manifest_df["en"] = [
                        normalize_text(ln)
                        for ln in read_txt_file(muavic_path / "en" / f"test_avsr.en")
                    ]
                    merged_trans = pd.merge(
                        left=manifest_df,
                        right=human_trans.drop_duplicates(subset=["en"]),
                        on="en",
                        how="left",
                    ).set_index(manifest_df.index)
                    assert len(merged_trans) == len(
                        manifest_df
                    ), "Mismatch between pseudo and human translation!!"
                    # select only the found matches
                    matched_indices = merged_trans[merged_trans[lang].notna()].index
                    merged_trans = merged_trans.loc[matched_indices]
                    # write out the results
                    write_txt_file(merged_trans[lang].tolist(), out_tgt_filepath)
                    merged_trans.drop([lang], axis=1, inplace=True)
                    write_av_manifest(merged_trans, out_manifest_filepath)
