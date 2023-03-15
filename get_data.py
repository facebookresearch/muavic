# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from pathlib import Path

from mtedx_utils import *
from lrs3_utils import *


def prepare_mtedx(args):
    # download mTEDx-{src_lang} files
    download_mtedx_data(args["mtedx"], args["src_lang"], args["src_lang"])
    if args["src_lang"] not in {"ar", "de"}:
        download_mtedx_data(args["mtedx"], args["src_lang"], "en")

    # download mTEDx videos
    download_mtedx_lang_videos(args["mtedx"], args["src_lang"])

    # pre-process audio files
    preprocess_mtedx_audio(args["mtedx"], args["src_lang"], args["muavic"])

    # pre-process video files
    preprocess_mtedx_video(
        args["mtedx"], args["metadata"], args["src_lang"], args["muavic"]
    )

    # prepare manifest
    prepare_mtedx_avsr_manifests(args["mtedx"], args["src_lang"], args["muavic"])

    # prepare translation
    if args["src_lang"] not in {"ar", "de"}:
        prepare_mtedx_avst_manifests(
            args["mtedx"], args["mt_trans"], args["src_lang"], args["muavic"]
        )


def prepare_lrs3(args):
    if is_empty(args["lrs3"]):
        print(
            "You have to download LRS3 dataset manually from this link:\n"
            + "https://mmai.io/datasets/lip_reading/\n"
            + "After downloading, decompress and place it in this directory: "
            + f"{args['root_path']}/lrs3"
        )
        return
    # segment LRS3 pretrain set
    segment_pretrain_videos_and_text(args["lrs3"])

    # process LRS3 videos
    process_lrs3_videos(args["lrs3"], args["metadata"], args["muavic"])

    # prepare manifest
    prepare_lrs3_avsr_manifests(args["lrs3"], args["muavic"])

    # prepare translation
    download_ted2020(args["ted2020"])
    prepare_lrs3_avst_manifests(args["mt_trans"], args["ted2020"], args["muavic"])


def main(args):
    # created needed directories
    dirs = ["muavic", "mtedx", "ted2020", "metadata", "mt_trans", "lrs3"]
    for dirname in dirs:
        args[dirname] = args["root_path"] / dirname
        args[dirname].mkdir(parents=True, exist_ok=True)

    # start creating MuAViC
    if args["src_lang"] == "en":
        # preapre LRS3 data
        prepare_lrs3(args)
    else:
        # Prepare mTEDx data
        prepare_mtedx(args)

    # clear out un-needed directories
    shutil.rmtree(args["mt_trans"])
    shutil.rmtree(args["metadata"])

    # We are finished!
    print(f"Creating MuAViC-{args['src_lang']} is completed!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-path",
        required=True,
        type=Path,
        help="Relative/Absolute path where MuAViC dataset will be downloaded.",
    )
    parser.add_argument(
        "--src-lang",
        required=True,
        choices=["ar", "de", "el", "en", "es", "fr", "it", "pt", "ru"],
        help="The language code for the source language in MuAViC.",
    )
    parser.add_argument(
        "--num-workers",
        default=os.cpu_count(),
        help="Max number of workers to be used in parallel.",
    )

    args = vars(parser.parse_args())
    main(args)
