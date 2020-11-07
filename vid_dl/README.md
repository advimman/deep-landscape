# Description
All videos are at least 1080p with CC license and are uploaded to Youtube from 28 Dec 2019 to 29 Jan 2020. 

`raw` - videos downloaded from youtube

`processed` - videos central cropped, resized to 1080x1080. Each video is 30FPS and has 100 frames. Each class has 23 videos: `{'landscape': 23, 'urban': 23, 'sky': 23}`

# Files
`sky_valid_23.json` contains video urls and timestamps of clips

`main.py` is a script to create this dataset (download all videos, trim to specidifc timestamps, central crop and resize them)

`create_images.py` is a script to split videos into frames.
