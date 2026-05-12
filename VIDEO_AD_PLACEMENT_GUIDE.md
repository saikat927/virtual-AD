# Video Ad Placement Guide

This document explains how to use and modify the `video_ad_placement.py` script for inserting virtual advertisements into football match videos.

## 1. Usage Instructions

Run the script from the command line using the following arguments:

```bash
python video_ad_placement.py --video "path/to/video.mp4" --logo "path/to/logo.png" [OPTIONS]
```

### Command Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--video` | String | (Required) | Path to the input video file. |
| `--logo` | String | (Required) | Path to the logo/ad image (supports PNG with alpha). |
| `--output` | String | `output_ad.mp4` | Path for the processed output video. |
| `--start_frame` | Int | `0` | First frame to process. |
| `--end_frame` | Int | `-1` | Last frame to process (-1 for end of video). |
| `--ad_x` | Float | `0.0` | X-coordinate of the ad center (meters). |
| `--ad_y` | Float | `25.0` | Y-coordinate of the ad center (meters). |
| `--ad_w` | Float | `10.0` | Width of the ad (meters). |
| `--ad_h` | Float | `2.0` | Height/Depth of the ad (meters). |

---

## 2. World Coordinate System

The script uses a **World Meter** coordinate system where $(0,0)$ is the **Center Spot** of the pitch.

- **Pitch Dimensions:** Assumed to be **105m x 68m**.
- **X-axis (Goal-to-Goal):** Ranges from **-52.5m** (Left Goal) to **+52.5m** (Right Goal).
- **Y-axis (Touchline-to-Touchline):** Ranges from **-34m** (Top) to **+34m** (Bottom).

### Example Placements:
- **Center Circle:** `--ad_x 0 --ad_y 0`
- **Near Top Touchline:** `--ad_y -25` (Negative is "up" on a standard broadcast view)
- **Near Bottom Touchline:** `--ad_y 25`
- **Behind Right Goal:** `--ad_x 55 --ad_y 0` (Assuming space exists outside the 52.5m line)

---

## 3. Code Architecture & Logic

The `VideoAdPlacement` class handles the heavy lifting through several key steps:

### A. Field Calibration (`get_homography`)
- Uses **HRNet** models to detect keypoints and lines on the pitch.
- Calculates a **Homography Matrix ($H_{w2i}$)** that maps World Meters to Image Pixels.
- **Optimization:** If a frame fails to calibrate, it reuses the homography from the previous frame to prevent "flickering" or missing ads.
- **Temporal Smoothing:** A moving average buffer (`h_buffer`) of size 5 is used to smooth out small camera jitters.

### B. Player Occlusion (`get_player_mask`)
- Uses a **Unet (ResNet34 encoder)** to segment players.
- The resulting mask ensures the advertisement appears **under** the players' feet rather than being drawn on top of them.

### C. Ad Insertion (`process_frame`)
1. **Logo-to-World:** Maps the flat logo image to the specified rectangle in World Meters.
2. **Logo-to-Image:** Combines the Logo-to-World and World-to-Image homographies.
3. **Perspective Warp:** Warps the logo into the camera's perspective.
4. **Composition:** Merges the warped logo with the original frame using the player mask and the logo's alpha channel.

---

## 4. Maintenance & Modifications

### To change smoothing intensity:
Modify `self.buffer_size = 5` in the `__init__` method. A larger number makes the ad more stable but slower to react to fast pans.

### To adjust calibration sensitivity:
In `get_homography`, you can modify the `threshold` values for `kp_dict` (Keypoints) and `lines_dict` (Lines). Lowering them detects more points but may increase noise.

### To swap models:
The script expects `SV_kp`, `SV_lines`, and `last_model.pt`. If you retrain these, simply update the file paths in the `main()` function.
