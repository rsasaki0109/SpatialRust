# OpenCV calibration comparison

Build/install the Python extension and run:

```bash
python bench/opencv_calibration_comparison/run.py \
  --output target/opencv-comparison/calibration.json
```

The suite uses OpenCV projection/fisheye conventions to synthesize observations,
then gates SpatialRust pinhole and Kannala–Brandt4 parameter recovery.
