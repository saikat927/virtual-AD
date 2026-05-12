# Project Summary: No Bells, Just Whistles (NBJW)

## Overview
This project implements a robust framework for **Sports Field Registration** in soccer broadcasts. It focuses on mapping 2D image coordinates to a 3D field model by leveraging geometric properties and camera calibration.

## Key Components
- **Inference (`inference.py`)**: Runs the registration pipeline on images or video files.
- **Deep Learning Models (`model/`)**: Uses HRNet-based architectures for detecting field keypoints and line segments.
- **Calibration (`sn_calibration/`)**: Implements 3D camera calibration using the DLT algorithm and SoccerNet annotations.
- **Utilities (`utils/`)**: Provides geometric calculations, field line extraction, and visualization tools.

## Supported Tasks
1. **Single-View Calibration**: Traditional main-camera registration.
2. **Multi-View Calibration**: Extending registration to multiple broadcast camera angles.
3. **Homography Estimation**: Comparative evaluation against state-of-the-art 2D mapping techniques.

## Usage
The system can be executed via `inference.py` for demos or through the provided shell scripts in `scripts/` for full dataset evaluation.
