
import math
from dataclasses import dataclass
from typing import List, Optional, Any

import cv2
import numpy as np
import mediapipe as mp
from moviepy import VideoFileClip

from models import FrameFeatures


@dataclass
class PerceptionConfig:
    """
    Configuration for feature extraction.

    Coordinates for zones and lid center are in normalized [0, 1] image space.
    """

    target_fps: float = 5.0

    workzone_x_min: float = 0.2
    workzone_x_max: float = 0.8
    workzone_y_min: float = 0.25
    workzone_y_max: float = 0.85

    toolzone_y_max: float = 0.25

    tool_roi_x_min: float = 0.25
    tool_roi_x_max: float = 0.75
    tool_roi_y_max: float = 0.60

    lid_center_x: float = 0.5
    lid_center_y: float = 0.5
    auto_estimate_lid_circle: bool = True
    lid_estimation_frames: int = 20
    lid_hough_dp: float = 1.2
    lid_hough_param1: int = 120
    lid_hough_param2: int = 35
    lid_min_radius_frac: float = 0.25
    lid_max_radius_frac: float = 0.49

    center_radius_frac: float = 0.18
    rim_inner_radius_frac: float = 0.30
    rim_outer_radius_frac: float = 0.48

    audio_window: float = 0.20

    audio_low_hz: float = 400.0
    audio_high_hz: float = 2000.0
    audio_high_max_hz: float = 8000.0

    hand_scale_min: float = 1e-3

    handle_roi_x_min: float = 0.72
    handle_roi_x_max: float = 0.95
    handle_roi_y_min: float = 0.22
    handle_roi_y_max: float = 0.55
    handle_dark_margin: int = 40
    handle_dark_ratio_strong: float = 0.10

    tool_center_roi_size_frac: float = 0.18


def _estimate_lid_circle(
    cap: cv2.VideoCapture,
    cfg: PerceptionConfig,
) -> Optional[tuple[float, float, float]]:
    """
    Estimate lid center (normalized x,y) and lid radius (pixels) using HoughCircles
    over the first few frames.

    Returns (cx_norm, cy_norm, r_pix) or None if detection fails.
    """
    pos0 = cap.get(cv2.CAP_PROP_POS_FRAMES)  # preserve caller's frame position

    centers = []
    radii = []
    used_frames = 0

    for _ in range(cfg.lid_estimation_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        used_frames += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        h, w = gray.shape

        x1 = int(cfg.workzone_x_min * w)
        x2 = int(cfg.workzone_x_max * w)
        y1 = int(cfg.workzone_y_min * h)
        y2 = int(cfg.workzone_y_max * h)

        roi = gray[y1:y2, x1:x2]  # limit search to workzone to avoid spurious circles
        if roi.size == 0:
            continue

        rh, rw = roi.shape
        min_dim = float(min(rh, rw))

        min_dim = float(min(h, w))
        min_r = int(cfg.lid_min_radius_frac * min_dim)
        max_r = int(cfg.lid_max_radius_frac * min_dim)

        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=cfg.lid_hough_dp,
            minDist=min_dim * 0.25,
            param1=cfg.lid_hough_param1,
            param2=max(18, int(cfg.lid_hough_param2 * 0.8)),
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is None:
            continue

        circles = circles[0]
        c = circles[np.argmax(circles[:, 2])]
        cx, cy, r = float(c[0]), float(c[1]), float(c[2])

        cx_full = cx + float(x1)
        cy_full = cy + float(y1)

        centers.append((cx_full / float(w), cy_full / float(h)))  # normalize to frame size
        radii.append(r)


    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)  # rewind capture to start

    if len(radii) < 2:
        return None

    cx_med = float(np.median([c[0] for c in centers]))
    cy_med = float(np.median([c[1] for c in centers]))
    r_med = float(np.median(radii))
    return cx_med, cy_med, r_med


def _load_audio_signal(video_path: str) -> tuple[Optional[np.ndarray], Optional[int]]:
    """
    Extract mono audio samples and sample rate from the video.

    Returns (audio_mono, sr). If there is no audio, returns (None, None).
    """
    clip = VideoFileClip(video_path)
    audio = clip.audio
    if audio is None:
        clip.close()
        return None, None

    sr = int(audio.fps)
    samples = audio.to_soundarray(fps=sr)
    clip.close()

    if samples.ndim == 2:
        audio_mono = samples.mean(axis=1)
    else:
        audio_mono = samples.astype(np.float32)

    return audio_mono, sr


def _audio_rms_at_time(
    audio: Optional[np.ndarray],
    sr: Optional[int],
    t: float,
    window: float,
) -> float:
    """
    Compute RMS energy of the audio around time t using a symmetric window.

    If audio is None, returns 0.0.
    """
    if audio is None or sr is None:
        return 0.0

    half_w = int(max(1, (window * sr) / 2))
    center = int(t * sr)
    start = max(0, center - half_w)
    end = min(len(audio), center + half_w)
    if end <= start:
        return 0.0

    segment = audio[start:end]
    return float(np.sqrt(np.mean(segment.astype(np.float32) ** 2)))

def _audio_band_rms_at_time(
    audio: Optional[np.ndarray],
    sr: Optional[int],
    t: float,
    window: float,
    f_lo: float,
    f_hi: float,
) -> float:
    """
    Compute a crude band-limited RMS proxy using an FFT on a short window.
    Returns 0.0 if audio is missing.
    """
    if audio is None or sr is None:
        return 0.0

    half_w = int(max(1, (window * sr) / 2))
    center = int(t * sr)
    start = max(0, center - half_w)
    end = min(len(audio), center + half_w)
    if end <= start + 8:
        return 0.0

    x = audio[start:end].astype(np.float32)

    win = np.hanning(len(x)).astype(np.float32)
    xw = x * win

    spec = np.fft.rfft(xw)  # short FFT as rough band energy proxy
    power = (np.abs(spec) ** 2).astype(np.float32)

    freqs = np.fft.rfftfreq(len(xw), d=1.0 / float(sr)).astype(np.float32)
    f_hi_eff = min(float(f_hi), float(sr) / 2.0)

    mask = (freqs >= float(f_lo)) & (freqs < f_hi_eff)
    if not np.any(mask):
        return 0.0

    band_energy = float(np.mean(power[mask]))
    return float(np.sqrt(max(0.0, band_energy)))


def _safe_norm(v: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.sqrt(float(np.sum(v * v))) + eps)


def _unit(v: np.ndarray) -> np.ndarray:
    n = _safe_norm(v)
    return v / n


def _hand_shape_features(
    lms: List[Any],
    scale_min: float,
) -> tuple[float, float, tuple[float, float, float]]:
    """
    Returns (openness, pinch, palm_normal_xyz).
    """
    WRIST = 0
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    PINKY_MCP = 17

    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    def vec(i: int) -> np.ndarray:
        lm = lms[i]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    wrist = vec(WRIST)
    middle_mcp = vec(MIDDLE_MCP)
    scale = max(scale_min, _safe_norm(middle_mcp - wrist))

    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    d_sum = 0.0
    for i in tips:
        d_sum += _safe_norm(vec(i) - wrist)
    openness = float((d_sum / float(len(tips))) / scale)

    pinch = float(_safe_norm(vec(THUMB_TIP) - vec(INDEX_TIP)) / scale)

    a = vec(INDEX_MCP) - wrist
    b = vec(PINKY_MCP) - wrist
    n = np.cross(a, b).astype(np.float32)
    n_u = _unit(n)

    return openness, pinch, (float(n_u[0]), float(n_u[1]), float(n_u[2]))

def _compute_optical_flow_features(
    prev_gray: Optional[np.ndarray],
    gray: np.ndarray,
    cx: float,
    cy: float,
    r_center: float,
    r_rim_in: float,
    r_rim_out: float,
) -> tuple[float, float, float, float, float, float]:
    """
    Compute mean motion magnitude and rotational index around (cx, cy),
    both globally and split into a center region and a rim band.

    cx, cy, and radii in pixel coordinates.
    """
    if prev_gray is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    h, w = gray.shape
    step = 4
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    u = flow[0:h:step, 0:w:step, 0]
    v = flow[0:h:step, 0:w:step, 1]

    mag = np.sqrt(u ** 2 + v ** 2)
    motion_mag = float(np.mean(mag))

    ru = xs - cx
    rv = ys - cy
    rad = np.sqrt(ru ** 2 + rv ** 2) + 1e-6

    r_hat_x = ru / rad
    r_hat_y = rv / rad
    t_hat_x = -r_hat_y
    t_hat_y = r_hat_x

    t_comp = np.abs(u * t_hat_x + v * t_hat_y)
    r_comp = np.abs(u * r_hat_x + v * r_hat_y)

    def rot_index_for_mask(mask: np.ndarray) -> float:
        if mask is None or mask.sum() == 0:
            return 0.0
        sum_t = float(np.sum(t_comp[mask]))
        sum_r = float(np.sum(r_comp[mask]))
        return float(sum_t / (sum_t + sum_r + 1e-6))  # tangential vs radial motion balance

    def mean_mag_for_mask(mask: np.ndarray) -> float:
        if mask is None or mask.sum() == 0:
            return 0.0
        return float(np.mean(mag[mask]))

    center_mask = rad <= r_center
    rim_mask = (rad >= r_rim_in) & (rad <= r_rim_out)

    rot_index = rot_index_for_mask(np.ones_like(center_mask, dtype=bool))  # global rotation index
    motion_center = mean_mag_for_mask(center_mask)
    rot_center = rot_index_for_mask(center_mask)
    motion_rim = mean_mag_for_mask(rim_mask)
    rot_rim = rot_index_for_mask(rim_mask)

    return motion_mag, rot_index, motion_center, rot_center, motion_rim, rot_rim


def _compute_hand_features(
    rgb_frame: np.ndarray,
    hands_detector: "mp.solutions.hands.Hands",
    cfg: PerceptionConfig,
) -> tuple[
    int, float, float, float, float, float, bool, bool,
    float, float, float, float,
    tuple[float, float, float], tuple[float, float, float]
]:
    """
    Run MediaPipe Hands and compute:

    - n_hands
    - left_x, left_y, right_x, right_y (normalized)
    - hand_dist
    - hands_in_workzone
    - hands_in_toolzone
    - left_openness, right_openness
    - left_pinch, right_pinch
    - left_palm_normal (nx,ny,nz), right_palm_normal (nx,ny,nz)
      Missing normals are (-1,-1,-1).
    """
    results = hands_detector.process(rgb_frame)

    left_x = left_y = -1.0
    right_x = right_y = -1.0
    n_hands = 0

    hands_in_workzone = False
    hands_in_toolzone = False

    left_open = right_open = -1.0
    left_pinch = right_pinch = -1.0
    left_n = (-1.0, -1.0, -1.0)
    right_n = (-1.0, -1.0, -1.0)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            label = handedness.classification[0].label
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))

            if label == "Left":
                left_x, left_y = cx, cy
            else:
                right_x, right_y = cx, cy

            if (
                cfg.workzone_x_min <= cx <= cfg.workzone_x_max
                and cfg.workzone_y_min <= cy <= cfg.workzone_y_max
            ):
                hands_in_workzone = True
            if cy <= cfg.toolzone_y_max:
                hands_in_toolzone = True

            open_v, pinch_v, n_xyz = _hand_shape_features(
                hand_landmarks.landmark,
                scale_min=cfg.hand_scale_min,
            )
            if label == "Left":
                left_open = open_v
                left_pinch = pinch_v
                left_n = n_xyz
            else:
                right_open = open_v
                right_pinch = pinch_v
                right_n = n_xyz

        n_hands = len(results.multi_hand_landmarks)

    if left_x >= 0 and right_x >= 0:
        dx = left_x - right_x
        dy = left_y - right_y
        hand_dist = float(math.sqrt(dx * dx + dy * dy))
    else:
        hand_dist = -1.0

    return (
        n_hands,
        left_x, left_y,
        right_x, right_y,
        hand_dist,
        hands_in_workzone,
        hands_in_toolzone,
        left_open, right_open,
        left_pinch, right_pinch,
        left_n, right_n,
    )


def _compute_tool_score(
    bgr_frame: np.ndarray,
    cfg: PerceptionConfig,
) -> float:

    h, w, _ = bgr_frame.shape

    x_min = int(cfg.tool_roi_x_min * w)
    x_max = int(cfg.tool_roi_x_max * w)
    y_min = 0
    y_max = int(cfg.tool_roi_y_max * h)

    roi = bgr_frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return 0.0

    roi_h, roi_w = roi.shape[:2]
    area = float(max(1, roi_h * roi_w))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_ratio = float(np.mean(edges > 0))
    edge_score = min(1.0, edge_ratio / 0.035)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=55,
        minLineLength=int(0.18 * roi_h),
        maxLineGap=14,
    )

    long_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length >= 0.18 * roi_h:
                long_lines += 1

    line_score = min(1.0, long_lines / 3.0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    mask1 = (h_ch <= 10) & (s_ch >= 70) & (v_ch >= 50)
    mask2 = (h_ch >= 160) & (s_ch >= 70) & (v_ch >= 50)
    red_mask = mask1 | mask2

    red_ratio = float(np.sum(red_mask)) / area
    red_score = min(1.0, red_ratio / 0.03)

    score = (0.45 * line_score) + (0.40 * edge_score) + (0.15 * red_score)  # combine shape + red hue cues
    return float(np.clip(score, 0.0, 1.0))

def _compute_handle_table_score(bgr_frame: np.ndarray, cfg: PerceptionConfig) -> float:
    """
    0..1 score for whether the handle is still sitting on the table
    in the top-right ROI. Uses an adaptive dark-pixel ratio.
    """
    h, w, _ = bgr_frame.shape
    x1 = int(cfg.handle_roi_x_min * w)
    x2 = int(cfg.handle_roi_x_max * w)
    y1 = int(cfg.handle_roi_y_min * h)
    y2 = int(cfg.handle_roi_y_max * h)

    roi = bgr_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)

    med = float(np.median(gray))
    thr = max(0.0, med - float(cfg.handle_dark_margin))  # adaptive darkness threshold for black handle

    dark_ratio = float(np.mean(gray < thr))
    score = dark_ratio / max(1e-6, cfg.handle_dark_ratio_strong)  # normalize so "on table" ~1
    return float(np.clip(score, 0.0, 1.0))


def _compute_tool_center_score(bgr_frame: np.ndarray, cfg: PerceptionConfig) -> float:
    """
    0..1 score for tool evidence near the lid center.
    Distinguish "tool lying on the table" vs "tool engaged at screw".
    """
    h, w, _ = bgr_frame.shape
    min_dim = float(min(h, w))
    half = int(0.5 * cfg.tool_center_roi_size_frac * min_dim)

    cx = int(cfg.lid_center_x * w)
    cy = int(cfg.lid_center_y * h)

    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)

    roi = bgr_frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # tool metal edges near screw head

    edge_ratio = float(np.mean(edges > 0))
    edge_score = min(1.0, edge_ratio / 0.05)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=int(0.35 * roi.shape[0]),
        maxLineGap=10,
    )

    long_lines = 0
    if lines is not None:
        for x1_, y1_, x2_, y2_ in lines[:, 0]:
            dx = x2_ - x1_
            dy = y2_ - y1_
            length = math.sqrt(dx * dx + dy * dy)
            if length >= 0.35 * roi.shape[0]:
                long_lines += 1

    line_score = min(1.0, long_lines / 2.0)  # strong lines imply tool aligned at center

    score = 0.55 * line_score + 0.45 * edge_score
    return float(np.clip(score, 0.0, 1.0))


def extract_frame_features(
    video_path: str,
    cfg: Optional[PerceptionConfig] = None,
) -> List[FrameFeatures]:
    """
    Main entry point.

    Reads the video at `video_path`, samples frames at cfg.target_fps,
    and returns a list of FrameFeatures objects.
    """
    if cfg is None:
        cfg = PerceptionConfig()

    audio, sr = _load_audio_signal(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = cfg.target_fps

    frame_interval = max(1, int(round(fps / cfg.target_fps)))

    lid_radius_pix: Optional[float] = None
    if cfg.auto_estimate_lid_circle:
        est = _estimate_lid_circle(cap, cfg)
        if est is not None:
            cfg.lid_center_x, cfg.lid_center_y, lid_radius_pix = est

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    temp_records = []
    prev_gray = None
    frame_idx = 0

    prev_sample_t: Optional[float] = None  # wall-clock of last sampled frame

    prev_left_xy: Optional[tuple[float, float]] = None
    prev_right_xy: Optional[tuple[float, float]] = None

    prev_left_n: Optional[np.ndarray] = None
    prev_right_n: Optional[np.ndarray] = None


    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        t = frame_idx / fps

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        cx_pix = cfg.lid_center_x * w
        cy_pix = cfg.lid_center_y * h

        if lid_radius_pix is not None and lid_radius_pix > 0:
            r_center = cfg.center_radius_frac * lid_radius_pix
            r_rim_in = cfg.rim_inner_radius_frac * lid_radius_pix
            r_rim_out = cfg.rim_outer_radius_frac * lid_radius_pix
        else:
            min_dim = float(min(h, w))
            r_center = cfg.center_radius_frac * min_dim
            r_rim_in = cfg.rim_inner_radius_frac * min_dim
            r_rim_out = cfg.rim_outer_radius_frac * min_dim

        (
            motion_mag,
            rot_index,
            motion_center,
            rot_center,
            motion_rim,
            rot_rim,
        ) = _compute_optical_flow_features(prev_gray, gray, cx_pix, cy_pix, r_center, r_rim_in, r_rim_out)

        prev_gray = gray


        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        (
            n_hands,
            left_x,
            left_y,
            right_x,
            right_y,
            hand_dist,
            hands_in_workzone,
            hands_in_toolzone,
            left_open,
            right_open,
            left_pinch,
            right_pinch,
            left_n_xyz,
            right_n_xyz,
        ) = _compute_hand_features(frame_rgb, mp_hands, cfg)

        dt_sample = 0.0 if prev_sample_t is None else float(t - prev_sample_t)  # sec since last sampled frame
        if dt_sample <= 1e-6:
            left_speed = -1.0
            right_speed = -1.0
            left_rot_speed = -1.0
            right_rot_speed = -1.0
        else:
            if left_x >= 0 and left_y >= 0 and prev_left_xy is not None:
                dx = left_x - prev_left_xy[0]
                dy = left_y - prev_left_xy[1]
                left_speed = float(math.sqrt(dx * dx + dy * dy) / dt_sample)
            else:
                left_speed = -1.0

            if right_x >= 0 and right_y >= 0 and prev_right_xy is not None:
                dx = right_x - prev_right_xy[0]
                dy = right_y - prev_right_xy[1]
                right_speed = float(math.sqrt(dx * dx + dy * dy) / dt_sample)
            else:
                right_speed = -1.0

            def rot_speed(prev_n: Optional[np.ndarray], cur_xyz: tuple[float, float, float]) -> float:
                if cur_xyz[0] < -0.5:
                    return -1.0
                cur = np.array(cur_xyz, dtype=np.float32)
                cur_norm = float(np.linalg.norm(cur))
                if cur_norm <= 1e-6:
                    return -1.0
                cur = cur / cur_norm
                if prev_n is None:
                    return -1.0
                dot = float(np.clip(float(np.dot(prev_n, cur)), -1.0, 1.0))
                ang = float(np.arccos(dot))  # angular change of palm normal
                return float(ang / dt_sample)

            left_rot_speed = rot_speed(prev_left_n, left_n_xyz)
            right_rot_speed = rot_speed(prev_right_n, right_n_xyz)

        prev_sample_t = float(t)

        if left_x >= 0 and left_y >= 0:
            prev_left_xy = (float(left_x), float(left_y))
            if left_n_xyz[0] >= -0.5:
                ln = np.array(left_n_xyz, dtype=np.float32)
                nrm = float(np.linalg.norm(ln))
                prev_left_n = ln / nrm if nrm > 1e-6 else None
            else:
                prev_left_n = None
        else:
            prev_left_xy = None
            prev_left_n = None

        if right_x >= 0 and right_y >= 0:
            prev_right_xy = (float(right_x), float(right_y))
            if right_n_xyz[0] >= -0.5:
                rn = np.array(right_n_xyz, dtype=np.float32)
                nrm = float(np.linalg.norm(rn))
                prev_right_n = rn / nrm if nrm > 1e-6 else None
            else:
                prev_right_n = None
        else:
            prev_right_xy = None
            prev_right_n = None


        dists = []
        cx_n = cfg.lid_center_x
        cy_n = cfg.lid_center_y
        if left_x >= 0 and left_y >= 0:
            dists.append(math.sqrt((left_x - cx_n) ** 2 + (left_y - cy_n) ** 2))
        if right_x >= 0 and right_y >= 0:
            dists.append(math.sqrt((right_x - cx_n) ** 2 + (right_y - cy_n) ** 2))
        hand_to_center = float(min(dists)) if dists else -1.0

        tool_score = _compute_tool_score(frame_bgr, cfg)
        tool_present = tool_score >= 0.60

        handle_table_score = _compute_handle_table_score(frame_bgr, cfg)
        tool_center_score = _compute_tool_center_score(frame_bgr, cfg)

        temp_records.append(
            {
                "t": t,
                "motion_mag": motion_mag,
                "rot_index": rot_index,
                "motion_center": motion_center,
                "rot_center": rot_center,
                "motion_rim": motion_rim,
                "rot_rim": rot_rim,
                "n_hands": n_hands,
                "left_x": left_x,
                "left_y": left_y,
                "right_x": right_x,
                "right_y": right_y,
                "hand_dist": hand_dist,
                "hand_to_center": hand_to_center,
                "hands_in_workzone": hands_in_workzone,
                "hands_in_toolzone": hands_in_toolzone,
                "tool_score": tool_score,
                "tool_present": tool_present,
                "handle_table_score": handle_table_score,
                "tool_center_score": tool_center_score,
                "left_speed": left_speed,
                "right_speed": right_speed,
                "left_openness": left_open,
                "right_openness": right_open,
                "left_pinch": left_pinch,
                "right_pinch": right_pinch,
                "left_rot_speed": left_rot_speed,
                "right_rot_speed": right_rot_speed,
            }
        )

        frame_idx += 1

    cap.release()
    mp_hands.close()

    features: List[FrameFeatures] = []
    if len(temp_records) == 0:
        return features

    audio_rms_values: List[float] = []
    audio_low_values: List[float] = []
    audio_high_values: List[float] = []

    for rec in temp_records:
        tt = float(rec["t"])
        audio_rms_values.append(_audio_rms_at_time(audio, sr, tt, cfg.audio_window))

        audio_low_values.append(
            _audio_band_rms_at_time(audio, sr, tt, cfg.audio_window, 0.0, cfg.audio_low_hz)
        )

        audio_high_values.append(
            _audio_band_rms_at_time(audio, sr, tt, cfg.audio_window, cfg.audio_high_hz, cfg.audio_high_max_hz)
        )

    audio_rms_values = np.array(audio_rms_values, dtype=np.float32)
    audio_low_values = np.array(audio_low_values, dtype=np.float32)
    audio_high_values = np.array(audio_high_values, dtype=np.float32)

    def norm01(x: np.ndarray) -> np.ndarray:
        if np.any(x > 0):
            m = float(x.max())
            if m > 0:
                return x / m
        return x

    audio_rms_values = norm01(audio_rms_values)
    audio_low_values = norm01(audio_low_values)
    audio_high_values = norm01(audio_high_values)

    audio_onset_values = np.zeros_like(audio_rms_values, dtype=np.float32)
    audio_low_onset = np.zeros_like(audio_low_values, dtype=np.float32)
    audio_high_onset = np.zeros_like(audio_high_values, dtype=np.float32)

    if len(audio_rms_values) >= 2:
        audio_onset_values[1:] = np.maximum(0.0, audio_rms_values[1:] - audio_rms_values[:-1])  # half-wave diff as onset
        audio_low_onset[1:] = np.maximum(0.0, audio_low_values[1:] - audio_low_values[:-1])
        audio_high_onset[1:] = np.maximum(0.0, audio_high_values[1:] - audio_high_values[:-1])

    for i, rec in enumerate(temp_records):
        ff = FrameFeatures(
            t=float(rec["t"]),
            lid_center_x=float(cfg.lid_center_x),
            lid_center_y=float(cfg.lid_center_y),

            motion_mag=float(rec["motion_mag"]),
            rot_index=float(rec["rot_index"]),
            motion_center=float(rec["motion_center"]),
            rot_center=float(rec["rot_center"]),
            motion_rim=float(rec["motion_rim"]),
            rot_rim=float(rec["rot_rim"]),

            n_hands=int(rec["n_hands"]),
            left_x=float(rec["left_x"]),
            left_y=float(rec["left_y"]),
            right_x=float(rec["right_x"]),
            right_y=float(rec["right_y"]),
            hand_dist=float(rec["hand_dist"]),
            hand_to_center=float(rec["hand_to_center"]),
            hands_in_workzone=bool(rec["hands_in_workzone"]),
            hands_in_toolzone=bool(rec["hands_in_toolzone"]),

            left_speed=float(rec.get("left_speed", -1.0)),
            right_speed=float(rec.get("right_speed", -1.0)),
            left_openness=float(rec.get("left_openness", -1.0)),
            right_openness=float(rec.get("right_openness", -1.0)),
            left_pinch=float(rec.get("left_pinch", -1.0)),
            right_pinch=float(rec.get("right_pinch", -1.0)),
            left_rot_speed=float(rec.get("left_rot_speed", -1.0)),
            right_rot_speed=float(rec.get("right_rot_speed", -1.0)),

            tool_score=float(rec["tool_score"]),
            tool_present=bool(rec["tool_present"]),

            audio_rms=float(audio_rms_values[i]),
            audio_onset=float(audio_onset_values[i]),
            audio_low_rms=float(audio_low_values[i]),
            audio_high_rms=float(audio_high_values[i]),
            audio_low_onset=float(audio_low_onset[i]),
            audio_high_onset=float(audio_high_onset[i]),

            handle_table_score=float(rec["handle_table_score"]),
            tool_center_score=float(rec["tool_center_score"]),
        )
        features.append(ff)

    return features
