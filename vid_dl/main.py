#!/usr/bin/env python3

import argparse
import os
import sys
import json
import traceback
import datetime
import glob
from pathlib import Path
import urllib.parse as urlparse
from urllib.parse import parse_qs

import cv2
import imageio
from tqdm import tqdm


TARGET_VID_SIZE = 1080  # resolution of target video
TARGET_VID_FPS = 30  # fps of target video
VIDEO_LENGTH = 100  # frames


def read_data():
    vids = json.load(open("sky_valid_23.json"))
    return vids


def crop_center(img, crop):
    y, x, c = img.shape
    startx = x//2-(crop//2)
    starty = y//2-(crop//2)
    return img[starty:starty+crop, startx:startx+crop, :]


def split_video_into_frames(path, start):
    """start - list of start seconds to cut video into subclips"""
    vid = imageio.get_reader(path,  'ffmpeg')
    metadata = vid.get_meta_data()
    fps = int(metadata['fps'])
    w = metadata["source_size"][0]
    h = metadata["source_size"][1]
    if (h < TARGET_VID_SIZE) or (w < TARGET_VID_SIZE):
        print(f"Low res video {os.path.split(path)}[-1]. Remove it from json.")
        os.remove(path)
        return None

    vid_frame_count = vid.count_frames()
    for i in range(len(start)-1):
        assert start[i]*fps + VIDEO_LENGTH < start[i+1]*fps, "Starts couldn't be so close"
    assert start[-1]*fps + VIDEO_LENGTH <= vid_frame_count, "Last start is too close to end"

    min_dim = min([h, w])

    frames = {k:[] for k in start}
    start_index = 0
    for i, frame in tqdm(enumerate(vid.iter_data()), total=start[-1]*fps+VIDEO_LENGTH):
        if i > start[-1]*fps+VIDEO_LENGTH:
            break

        if i > start[start_index]*fps+VIDEO_LENGTH:
            start_index += 1

        if (i >= start[start_index]*fps) and (i < start[start_index]*fps+VIDEO_LENGTH):
            frame = crop_center(frame, min_dim)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (TARGET_VID_SIZE, TARGET_VID_SIZE))
            frames[start[start_index]].append(frame)

    return frames


def save_video(frames, path):
    assert len(frames) == VIDEO_LENGTH, f"{path} x {len(frames)}"
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), TARGET_VID_FPS, (TARGET_VID_SIZE, TARGET_VID_SIZE))
    for frame in frames:
        writer.write(frame.astype('uint8'))
    writer.release()


def process_video(video_type, video_link, video_start):
    def _get_save_path(start_s):
        return os.path.join(PATH['processed'], f"{video_type}_{fname}_{start_s}.mp4")

    if "watch?v=" in video_link:
        fname = parse_qs(urlparse.urlparse(video_link).query)['v'][0]
    else:
        fname = video_link.split("/")[-1]
    video_path = os.path.join(PATH['raw'], f"{fname}.mp4")

    # Return if all starts already trimed and saved
    return_flag = []
    for start_s in video_start:
        return_flag.append(os.path.exists(_get_save_path(start_s)))
    if all(return_flag):
        return

    # process
    trimed_frames = split_video_into_frames(path=video_path, start=video_start)
    if trimed_frames is None:
        return

    for start_s, tfs in trimed_frames.items():
        save_video(frames=tfs, path=_get_save_path(start_s))
    return


def calculate_stats(vids):
    files = glob.glob(os.path.join(PATH['processed'], "*.mp4"))
    res = {k: 0 for k in vids.keys()}
    for f in files:
        video_type = os.path.split(f)[-1].split("_")[0]
        res[video_type] += 1
    print(f"Dataset stats: {res}")


def run():
    vids = read_data()
    for video_type in vids.keys():
        print(video_type, len(vids[video_type]))
        for link, video_start in vids[video_type].items():
            if "watch?v=" in link:
                parsed = urlparse.urlparse(link)
                name = parse_qs(parsed.query)['v'][0]
            else:
                name = link.split("/")[-1]
            path = os.path.join(PATH["raw"], f"{name}.mp4")

            try:
                # download videos
                if not os.path.exists(path):
                    duration = "{:0>8}".format(str(datetime.timedelta(seconds=max(video_start) + 20)))
                    os.system(f"ffmpeg -i $(youtube-dl -f 'bestvideo[height={TARGET_VID_SIZE}]' --get-url '{link}') -ss 00:00:00 -t {duration} -c:v copy -c:a copy {path}")

                # process_video
                if len(video_start) > 0:
                    process_video(video_type=video_type, video_link=link, video_start=video_start)
            except Exception:
                print("bad_video", f"{traceback.format_exc()}", link)
    calculate_stats(vids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    args = parser.parse_args()
    PATH = {"root": os.path.join(args.output_dir, "sky_valid_23")}
    PATH["raw"] = os.path.join(PATH['root'], "raw")
    PATH["processed"] = os.path.join(PATH['root'], "processed")
    for v in PATH.values():
        Path(v).mkdir(parents=True, exist_ok=True)
    run()
