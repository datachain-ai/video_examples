# Audio & Video Examples Tutorial

This tutorial shows how to process and analyze audio/video data at scale with DataChain.

Unlike big data (lots of rows in tables), **heavy data** is large, complex, unstructured
files - videos, audio, images - rich in information but harder to query directly.

DataChain turns heavy data into structured, queryable form for fast analysis and
integration with AI/ML pipelines, dashboards, and LLM reasoning.


📊 Why this matters:
- Turn unstructured heavy data into structured, analyzable form.
- Generate new features and signals for deeper insight.
- Process millions of files at high speed using parallel and distributed compute.

## Data Model Studio UI

![Data Model UI](assets/datamodel.gif)

## 1. Extract Frames from Video & Detect Objects

Install dependencies:
```shell
uv pip install -r requirements.txt
```

Run the Frame Extractor:
```shell
python video-detector.py
```
<details>
<summary>video-detector.py script</summary>

```python
# /// script
# dependencies = [
#   "datachain[video,audio]",
#	"opencv-python",
#	"ultralytics",
# ]
# ///

import os
from typing import Iterator

import datachain as dc
from datachain import VideoFile, ImageFile
from datachain.model.ultralytics import YoloBBoxes, YoloSegments, YoloPoses

from pydantic import BaseModel
from ultralytics import YOLO, settings

local = False
bucket = "data-video" if local else "gs://datachain-starss23/"
input_path = f"{bucket}/balanced_train_segments/video"
output_path = f"{bucket}/temp/video-detector-frames"
detection_dataset = "frames-detector"
target_fps = 1

model_bbox = "yolo11n.pt"
model_segm = "yolo11n-seg.pt"
model_pose = "yolo11n-pose.pt"


# Upload models to avoid YOLO-downloader issues
if not local:
    weights_dir = f"{os.getcwd()}/{settings['weights_dir']}"
    dc.read_storage([
        f"{bucket}/models/{model_bbox}",
        f"{bucket}/models/{model_segm}",
        f"{bucket}/models/{model_pose}",
    ]
    ).to_storage(weights_dir, placement="filename")

    model_bbox = f"{weights_dir}/{model_bbox}"
    model_segm = f"{weights_dir}/{model_segm}"
    model_pose = f"{weights_dir}/{model_pose}"


class YoloDataModel(BaseModel):
    bbox: YoloBBoxes
    segm: YoloSegments
    poses: YoloPoses


class VideoFrameImage(ImageFile):
    num: int
    orig: VideoFile


def extract_frames(file: VideoFile) -> Iterator[VideoFrameImage]:
    info = file.get_info()

    # one frame per sec
    step = int(info.fps / target_fps) if target_fps else 1
    frames = file.get_frames(step=step)

    for num, frame in enumerate(frames):
        image = frame.save(output_path, format="jpg")
        yield VideoFrameImage(**image.model_dump(), num=num, orig=file)


def process_all(yolo: YOLO, yolo_segm: YOLO, yolo_pose: YOLO, frame: ImageFile) -> YoloDataModel:
    img = frame.read()
    return YoloDataModel(
        bbox=YoloBBoxes.from_results(yolo(img, verbose=False)),
        segm=YoloSegments.from_results(yolo_segm(img, verbose=False)),
        poses=YoloPoses.from_results(yolo_pose(img, verbose=False))
    )


def process_bbox(yolo: YOLO, frame: ImageFile) -> YoloBBoxes:
    return YoloBBoxes.from_results(yolo(frame.read(), verbose=False))


chain = (
    dc
    .read_storage(input_path, type="video")
    .filter(dc.C("file.path").glob("*.mp4"))
    .sample(2)
    .settings(parallel=5)

    .gen(frame=extract_frames)

    # Initialize models: once per processing thread
    .setup(
        yolo=lambda: YOLO(model_bbox),
        # yolo_segm=lambda: YOLO(model_segm),
        # yolo_pose=lambda: YOLO(model_pose)
    )

    # Apply yolo detector to frames
    .map(bbox=process_bbox)
    # .map(yolo=process_all)
    .order_by("frame.path", "frame.num")
    .save(detection_dataset)
)

if local:
    chain.show()
```

</details>

This script:
1. Loads videos from cloud storage
2. Extracts frames at 1fps (configurable)
3. Runs YOLO object detection, segmentation, and pose detection
4. Saves results to a DataChain dataset for analysis

### Extracted Frames with Humans

<img src="assets/humanframes.png" alt="Frames with humans" width="75%" style="border: 2px solid black;">

## 2. Filter for Videos with Humans

```shell
python video-humans.py
```

<details>
<summary>video-humans.py script</summary>

```python
# /// script
# dependencies = ["datachain"]
# ///

import datachain as dc
from datachain.func.array import contains

target_class = "person"
input_dataset = "frames-detector"
output_dataset = "detector-human-videos"

chain_humans = (
    dc.read_dataset(input_dataset)
    .filter(contains("bbox.name", target_class))

    # Select only signals that are required
    .mutate(file=dc.C("frame.orig"))
    .select("file")

    # Remove file duplicats
    .distinct("file")
    .save(output_dataset)
)
```

</details>

### Detection Results

| **Description** | **Image** |
|-------------|-------|
| **Correct detection (with people)** | <img src="assets/humanvideos.png" alt="Correct detection - with people" width="50%"> |
| **False detection (no people)** | <img src="assets/humanvideos-no-people.png" alt="False detection no people" width="50%"> |

Better detection (with people) using highest confidence score:

<img src="assets/humanvideos-confidence.png" alt="High confidence score" width="75%" style="border: 2px solid black;">

## 3. Extract Video Fragments

```shell
python video-fragments.py
```

<details>
<summary>video-fragments.py script</summary>

```python
# /// script
# dependencies = [
#   "datachain[video,audio]",
# ]
# ///

from typing import Iterator

import datachain as dc
from datachain import VideoFile, VideoFragment

local = False
bucket = "data-video" if local else "gs://datachain-starss23/"
input_path = f"{bucket}/balanced_train_segments/video"
output_path = f"{bucket}/temp/video-fragments"
fragments_dataset = "video-fragments"
segment_duration = 7


class VideoClip(VideoFile):
    orig: VideoFragment


def video_fragments(file: VideoFile) -> Iterator[VideoClip]:
    for fragment in file.get_fragments(segment_duration):
        clip = fragment.save(output_path)
        yield VideoClip(**clip.model_dump(), orig=fragment)


chain = (
    dc
    .read_storage(input_path, type="video")
    .filter(dc.C("file.path").glob("*.mp4"))
    .sample(10)
    .settings(parallel=5)

    .gen(clip=video_fragments)

    .order_by("clip.path", "clip.orig.start")
    .save(fragments_dataset)
)

if local:
    chain.show()
```

</details>

## 4. Add Noise to Videos

### Using OpenCV:
```shell
python video-noise.py
```

### Using FFmpeg:
```shell
python video-noise-ffmpeg.py
```

## 5. Generate Statistics

```shell
python video-stats.py
```

<details>
<summary>video-stats.py script</summary>

```python
# /// script
# dependencies = ["datachain"]
# ///

import datachain as dc
from datachain.func.array import contains

class_names = ["person", "handbag", "car", "truck"]
input_dataset = "frames-detector"
stats_dataset = "detector-stats"

chain = dc.read_dataset(input_dataset)

total_frames = chain.count()
total_videos = chain.distinct("frame.orig").count()

dc.read_values(
    class_name = class_names,
    frame_coverage = [
        chain.filter(contains("bbox.name", name)).count()*1.0/total_frames
        for name in class_names
    ],
    video_coverage = [
        chain.filter(contains("bbox.name", name)).distinct("frame.orig").count()*1.0/total_videos
        for name in class_names
    ],
).save(stats_dataset)
```

</details>

## 6. Audio Processing

### Convert FLAC to MP3:
```shell
python to_mp3.py
# or using pydub
python to_mp3_with_pydub.py
```

### Download to Local:
```shell
python to_local.py
```

### Generate Spectrograms:
```shell
python spectogram.py
```

### Audio Segmentation:
```shell
python segment.py
python segment_trim.py
python segment_stats.py
```

### Apply Bandpass Filter:
```shell
python bandpass.py
```

### Query Waveforms:
```shell
python waveform_query.py parquet_file filename channel [--info]
```

## Features

- **Parallel Processing**: Process millions of files using distributed compute
- **Type Safety**: Pydantic models for structured data
- **Cloud Native**: Works with S3, GCS, and local storage
- **ML Integration**: Built-in support for YOLO, librosa, and other ML libraries
- **Efficient**: Stream processing with minimal memory footprint