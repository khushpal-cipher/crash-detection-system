import numpy as np
import cv2
import torch
import torch.nn.functional as F

class DepthEstimator:
    """
    Wraps Intel MiDaS v3.1 (small variant) for monocular depth estimation.
    Supports Apple MPS GPU with CPU fallback.
    """
    _model = None
    _transform = None

    def __init__(self):
        self.enabled = False
        
        # Determine device (MPS, CUDA, or CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        try:
            # Load model and transform globally (cached)
            if DepthEstimator._model is None:
                DepthEstimator._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                DepthEstimator._transform = midas_transforms.small_transform
            
            self.model = DepthEstimator._model.to(self.device)
            self.model.eval()
            self.transform = DepthEstimator._transform
            self.enabled = True
        except Exception as e:
            print(f"Failed to load MiDaS model: {e}")
            self.enabled = False

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Estimates relative inverse-depth map for the input frame.
        
        Args:
            frame_bgr: BGR frame as numpy array (H, W, 3).
            
        Returns:
            Relative inverse-depth map (H, W) as float32 numpy array, or None if disabled.
        """
        if not self.enabled:
            return None

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform (resizes to 256x256 internally for small_transform)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            # Estimate depth
            prediction = self.model(input_batch)

            # Resize output to original frame dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert back to numpy array
        output = prediction.cpu().numpy().astype(np.float32)
        return output

    def estimate_metric(self, frame_bgr: np.ndarray, h_cam: float, pitch_deg: float, fy: float, cy: float) -> np.ndarray:
        """
        Estimates metric depth map in meters by calibrating MiDaS output against pinhole ground-plane geometry.
        
        Args:
            frame_bgr: BGR frame as numpy array (H, W, 3).
            h_cam: Camera height in meters.
            pitch_deg: Camera pitch angle in degrees (downward is positive).
            fy: Focal length in y direction.
            cy: Principal point y-coordinate.
            
        Returns:
            Metric depth map (H, W) in meters, clamped to [0.5, 100.0], or None if disabled.
        """
        midas_relative = self.estimate(frame_bgr)
        if midas_relative is None:
            return None

        H, W = midas_relative.shape
        pitch_rad = np.radians(pitch_deg)

        # Take the bottom 30% of the image (ground region)
        start_v = int(H * 0.7)
        ground_region = midas_relative[start_v:H, :]

        # Analytical pinhole depth for each row in ground region
        v = np.arange(start_v, H)
        
        # Z = h_cam / tan(pitch_rad + arctan((v - cy) / fy))
        angles = pitch_rad + np.arctan((v - cy) / fy)
        
        # Avoid division by zero and negative angles (sky pointing)
        valid_mask = angles > 0.01 
        
        analytical_depths = np.zeros_like(v, dtype=np.float32)
        analytical_depths[valid_mask] = h_cam / np.tan(angles[valid_mask])
        
        # Broadcast to shape (H_ground, W) to match midas_relative in the ground region
        analytical_depths_2d = np.tile(analytical_depths[:, np.newaxis], (1, W))

        # Compute scale factor using valid depths
        valid_analytical = analytical_depths_2d[valid_mask, :]
        valid_midas = ground_region[valid_mask, :]
        
        if valid_midas.size == 0 or np.all(valid_midas == 0):
             return np.clip(midas_relative, 0.5, 100.0) # Fallback if calibration fails

        # Handle zero divisions in midas
        safe_midas = np.where(valid_midas == 0, 1e-6, valid_midas)
        scale = np.median(valid_analytical / safe_midas)
        
        if np.isnan(scale) or np.isinf(scale) or scale <= 0:
             return np.clip(midas_relative, 0.5, 100.0)

        # Apply scale: metric_depth = midas_relative * scale
        metric_depth = midas_relative * scale

        # Clamp result to [0.5, 100.0] meters
        metric_depth = np.clip(metric_depth, 0.5, 100.0)
        return metric_depth

    def get_depth_at(self, metric_depth: np.ndarray, x: int, y: int) -> float:
        """
        Returns metric depth at pixel (x, y).
        
        Args:
            metric_depth: Metric depth map (H, W).
            x: x-coordinate.
            y: y-coordinate.
            
        Returns:
            Metric depth at (x, y) or None if metric_depth is None.
        """
        if metric_depth is None:
            return None
            
        H, W = metric_depth.shape
        if 0 <= y < H and 0 <= x < W:
            return float(metric_depth[y, x])
        return None

    def colorize(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Returns (H, W, 3) BGR heatmap visualization of the depth map.
        
        Args:
            depth_map: Depth map (relative or metric).
            
        Returns:
            BGR heatmap (H, W, 3) or None if disabled.
        """
        if depth_map is None:
            return None

        # Normalize depth map to 0-255
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max == depth_min:
            depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            depth_normalized = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)

        # Apply colormap (INFERNO is standard for depth)
        colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        return colormap
