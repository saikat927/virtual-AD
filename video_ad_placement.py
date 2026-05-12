import cv2
import numpy as np
import torch
import yaml
import argparse
import os
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# Import repository modules
from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l
from utils.utils_calib import FramebyFrameCalib
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict
import filters

from collections import deque

class VideoAdPlacement:
    def __init__(self, weights_kp, weights_line, weights_seg, device='cpu', buffer_size=15, h_ema_alpha=0.15):
        self.device = device
        print(f"Loading models onto {device}...")
        self.load_calibration_models(weights_kp, weights_line)
        self.load_segmentation_model(weights_seg)
        self.preprocess_seg = get_preprocessing_fn("resnet34", pretrained="imagenet")
        
        # Transformations for calibration models
        self.calib_transform = T.Compose([
            T.Resize((540, 960)),
            T.ToTensor()
        ])

        # Online Filtering Buffer (Deque)
        self.buffer_size = buffer_size
        self.cp_buffer = deque(maxlen=self.buffer_size)
        
        # Homography Smoothing
        self.prev_h = None
        self.h_ema_alpha = h_ema_alpha

    def load_calibration_models(self, weights_kp, weights_line):
        cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
        cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))
        
        self.model_kp = get_cls_net(cfg)
        self.model_kp.load_state_dict(torch.load(weights_kp, map_location=self.device))
        self.model_kp.to(self.device).eval()
        
        self.model_line = get_cls_net_l(cfg_l)
        self.model_line.load_state_dict(torch.load(weights_line, map_location=self.device))
        self.model_line.to(self.device).eval()

    def load_segmentation_model(self, weights_seg):
        self.model_seg = smp.Unet(
            encoder_name="resnet34",
            in_channels=4,
            classes=1,
        )
        checkpoint = torch.load(weights_seg, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model_seg.load_state_dict(state_dict)
        self.model_seg.to(self.device).eval()

    def color_seg_hsv(self, image):
        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define HSV range for green (may require tuning for your dataset)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        
        # Create mask to detect green
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Invert mask to get non-green regions (i.e., players, ball, ref)
        non_green_mask = cv2.bitwise_not(green_mask)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, 50, 150)
        
        non_green_mask = non_green_mask|canny
        
        return non_green_mask/255.0

    def get_cam_params(self, frame):
        h_orig, w_orig = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_input = self.calib_transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            heatmaps = self.model_kp(img_input)
            heatmaps_l = self.model_line(img_input)
        
        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
        kp_dict = coords_to_dict(kp_coords, threshold=0.1486)
        lines_dict = coords_to_dict(line_coords, threshold=0.3880)
        final_dict = complete_keypoints(kp_dict, lines_dict, w=960, h=540, normalize=True)

        cam = FramebyFrameCalib(iwidth=w_orig, iheight=h_orig, denormalize=True)
        cam.update(final_dict[0])
        best_res = cam.heuristic_voting()
        
        if best_res is None:
            return None

        return best_res["cam_params"]

    def cam_params_to_homography(self, cp):
        Q = np.array([[cp['x_focal_length'], 0, cp['principal_point'][0]], 
                      [0, cp['y_focal_length'], cp['principal_point'][1]], 
                      [0, 0, 1]])
        
        if 'rotation_matrix' in cp:
            R = np.array(cp['rotation_matrix'])
        else:
            pan = cp['pan_degrees'] * np.pi / 180.
            tilt = cp['tilt_degrees'] * np.pi / 180.
            roll = cp['roll_degrees'] * np.pi / 180.
            R = np.transpose(filters.pan_tilt_roll_to_orientation(pan, tilt, roll))

        It = np.eye(4)[:-1]
        It[:, -1] = -np.array(cp['position_meters'])
        P = Q @ (R @ It)
        h_w2i = P[:, [0, 1, 3]]
        h_w2i = h_w2i / h_w2i[2, 2] # Normalize
        return h_w2i

    def apply_filtering(self, params_list):
        if len(params_list) < 3:
            return params_list

        # 1. Handle missing parameters
        params_valid, is_erroneous, erroneous_pos = filters.to_valid_cam_params(params_list)
        params_type = filters.camParamsPerImage_to_camParamsPerType(params_valid)
        
        # 2. Linear Interpolation
        params_type = filters.linear_interpolation(params_type, is_erroneous, erroneous_pos)
        
        # 3. Median filter (ensure odd window size >= 3)
        win_med = min(5, len(params_list))
        if win_med % 2 == 0: win_med -= 1
        if win_med >= 3:
            params_type = filters.outliers_remover(params_type, is_erroneous, erroneous_pos, windowLength=win_med)
        
        # 4. Savitzky-Golay (ensure odd window size >= 3)
        window_sg = min(self.buffer_size, len(params_list))
        if window_sg % 2 == 0: window_sg -= 1
        if window_sg >= 3:
            params_type = filters.camParamsSmoothing(params_type, windowLength=window_sg)
            
        return filters.camParamsPerType_to_camParamsPerImage(params_type)

    def process_online(self, frame, logo, world_coords):
        """Processes frames using a sliding window for filtering."""
        # 1. Estimate current frame params and add to buffer
        cp = self.get_cam_params(frame)
        self.cp_buffer.append(cp)

        # 2. Apply filtering to the current buffer
        filtered_params = self.apply_filtering(list(self.cp_buffer))
        
        # 3. Use the LATEST frame's filtered parameters (index -1)
        best_cp = filtered_params[-1]
        
        if best_cp is None:
            return frame

        # 4. Render
        h_w2i = self.cam_params_to_homography(best_cp)
        
        # --- Direct Homography Smoothing (EMA) ---
        if h_w2i is not None:
            if self.prev_h is not None:
                # Apply EMA: alpha * current + (1 - alpha) * previous
                h_w2i = self.h_ema_alpha * h_w2i + (1 - self.h_ema_alpha) * self.prev_h
                # Re-normalize to ensure h22 = 1
                h_w2i = h_w2i / h_w2i[2, 2]
            self.prev_h = h_w2i
        # -----------------------------------------

        result = self.render_ad(frame, logo, world_coords, h_w2i)
        
        return result

    def get_player_mask(self, frame):
        h_orig, w_orig = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (512, 512))
        
        # Get non-green mask
        non_green_mask = self.color_seg_hsv(img_resized) # (512, 512)
        
        img_pre = self.preprocess_seg(img_resized) # (512, 512, 3)
        
        # Concatenate: (512, 512, 3) + (512, 512, 1) -> (512, 512, 4)
        img_4ch = np.concatenate([img_pre, non_green_mask[..., np.newaxis]], axis=-1)
        
        input_tensor = torch.tensor(img_4ch, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model_seg(input_tensor)
            mask = (output[0, 0] > 0).cpu().numpy().astype(np.uint8)
        
        return cv2.resize(mask, (w_orig, h_orig))

    def render_ad(self, frame, logo, world_coords, h_w2i):
        if h_w2i is None:
            return frame
            
        h_frame, w_frame = frame.shape[:2]
        h_logo, w_logo = logo.shape[:2]
        
        pts_logo = np.array([[0, 0], [w_logo, 0], [w_logo, h_logo], [0, h_logo]], dtype=np.float32)
        pts_world = np.array(world_coords, dtype=np.float32)
        H_l2w, _ = cv2.findHomography(pts_logo, pts_world)
        H_l2i = h_w2i @ H_l2w
        
        warped_logo = cv2.warpPerspective(logo, H_l2i, (w_frame, h_frame))
        player_mask = self.get_player_mask(frame)
        
        if logo.shape[2] == 4:
            alpha = logo[:,:,3] / 255.0
            warped_alpha = cv2.warpPerspective(alpha, H_l2i, (w_frame, h_frame))
            logo_mask = (warped_alpha > 0.5).astype(np.uint8)
        else:
            logo_mask = (np.max(warped_logo, axis=2) > 0).astype(np.uint8)
        
        final_mask = cv2.bitwise_and(logo_mask, cv2.bitwise_not(player_mask))
        final_mask_3ch = np.stack([final_mask]*3, axis=-1)
        
        result = frame.copy()
        mask_idx = final_mask_3ch > 0
        
        # Weighted sum: 20% of original frame and 80% of the ad image
        blended = cv2.addWeighted(frame, 0.3, warped_logo[..., :3], 0.7, 0)
        result[mask_idx] = blended[mask_idx]
        return result

def main():
    parser = argparse.ArgumentParser(description="Virtual Ad Placement for Video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--logo", type=str, required=True, help="Path to logo image (PNG/JPG)")
    parser.add_argument("--output", type=str, default="output_ad.mp4", help="Path to output video")
    parser.add_argument("--start_frame", type=int, default=0, help="Starting frame index")
    parser.add_argument("--end_frame", type=int, default=-1, help="Ending frame index (-1 for end of video)")
    
    # Ad Placement Variables (Meters)
    parser.add_argument("--ad_x", type=float, default=0.0)
    parser.add_argument("--ad_y", type=float, default=15.0)
    parser.add_argument("--ad_w", type=float, default=15.0)
    parser.add_argument("--ad_h", type=float, default=12.0)
    parser.add_argument("--buffer", type=int, default=15, help="Online filtering buffer size")
    parser.add_argument("--h_alpha", type=float, default=0.15, help="Homography smoothing factor (EMA)")

    args = parser.parse_args()

    half_w, half_h = args.ad_w / 2, args.ad_h / 2
    AD_REGION_WORLD = [
        [args.ad_x - half_w, args.ad_y - half_h],
        [args.ad_x + half_w, args.ad_y - half_h],
        [args.ad_x + half_w, args.ad_y + half_h],
        [args.ad_x - half_w, args.ad_y + half_h]
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manager = VideoAdPlacement("SV_kp", "SV_lines", "seg_player_ball_UNET88.pt", device, buffer_size=args.buffer, h_ema_alpha=args.h_alpha)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if args.end_frame == -1: args.end_frame = total_frames
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    logo = cv2.imread(args.logo, cv2.IMREAD_UNCHANGED)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    num_frames = args.end_frame - args.start_frame
    pbar = tqdm(total=num_frames, desc="Processing Online")
    
    frame_idx = args.start_frame
    while cap.isOpened() and frame_idx < args.end_frame:
        ret, frame = cap.read()
        if not ret: break
            
        result = manager.process_online(frame, logo, AD_REGION_WORLD)
        out.write(result)
        
        frame_idx += 1
        pbar.update(1)
        
    cap.release()
    out.release()
    pbar.close()
    print(f"Video saved to {args.output}")

if __name__ == "__main__":
    main()
