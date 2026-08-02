# OpenCV video comparison

This suite compares SpatialRust dense integer block flow with OpenCV Farneback
flow on a deterministic translated texture:

```bash
python bench/opencv_video_comparison/run.py \
  --output target/opencv-comparison/video.json
```
