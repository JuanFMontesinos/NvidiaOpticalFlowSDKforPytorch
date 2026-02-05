# Assets Directory

This directory contains sample image frames for testing and demonstrating the optical flow functionality.

## Current Assets

- `frame_0001.png` - First frame of a sample sequence
- `frame_0002.png` - Second frame of a sample sequence
- `frame_0001.flo` - Ground truth optical flow (if available)

## Using Your Own Images

To use your own images with the examples:

1. **Add your frames**: Place two consecutive frames in this directory
   ```bash
   cp /path/to/your/frame1.png assets/frame_0001.png
   cp /path/to/your/frame2.png assets/frame_0002.png
   ```

2. **Run the minimal example**:
   ```bash
   cd examples
   python minimal_example.py
   ```

3. **Check the output**: Results will be saved in the `output/` directory

## Image Requirements

- **Format**: PNG, JPG, or any format supported by imageio
- **Color**: RGB or RGBA (alpha channel will be handled automatically)
- **Data type**: uint8 (0-255) or float (0.0-1.0)
- **Resolution**: Any resolution supported by your GPU (tested up to 4K)

## Recommended Test Sequences

For testing optical flow algorithms, consider these publicly available datasets:

1. **MPI-Sintel**: http://sintel.is.tue.mpg.nl/
   - High-quality synthetic sequences with ground truth
   - Recommended for evaluation

2. **KITTI Vision Benchmark**: http://www.cvlibs.net/datasets/kitti/
   - Real-world driving sequences
   - Good for autonomous driving applications

3. **Middlebury Optical Flow**: https://vision.middlebury.edu/flow/
   - Classic benchmark dataset
   - Small sequences, good for quick testing

4. **Flying Chairs**: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html
   - Large-scale synthetic dataset for training

## Example: Download Sample Frames

Here's how to download some sample frames for testing:

```bash
# Download sample frames from MPI-Sintel (requires dataset download)
# Or use any video and extract frames:

# Using ffmpeg to extract frames from a video:
ffmpeg -i your_video.mp4 -vf "select='eq(n,0)'" -vframes 1 assets/frame_0001.png
ffmpeg -i your_video.mp4 -vf "select='eq(n,1)'" -vframes 1 assets/frame_0002.png

# Or using Python with imageio:
python << EOF
import imageio.v3 as imageio
video = imageio.imread("your_video.mp4")
imageio.imwrite("assets/frame_0001.png", video[0])
imageio.imwrite("assets/frame_0002.png", video[1])
EOF
```

## File Format Notes

### .flo Format (Ground Truth)
The `.flo` format is the standard Middlebury format for storing optical flow:
- Header: 4 bytes magic number (202021.25)
- Width: 4 bytes (int32)
- Height: 4 bytes (int32)
- Data: width × height × 2 × 4 bytes (float32)
  - Channel 0: horizontal displacement (x)
  - Channel 1: vertical displacement (y)

Use `of.io.read_flo()` and `of.io.write_flo()` to work with this format.
