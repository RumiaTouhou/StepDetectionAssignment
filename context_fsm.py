
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

import numpy as np

from models import FrameFeatures, ActionLabel, StepEvent, Procedure


class ContextState(Enum):
    IDLE = 0
    LID_REORIENT = 1
    PICK_WASHER = 2
    SEAT_WASHER = 3
    PICK_SCREW = 4
    INSERT_SCREW = 5
    PICK_HANDLE = 6
    INSERT_HANDLE = 7
    HAND_TIGHTEN_HANDLE = 8
    PLACE_LID_FOR_TOOL = 9
    PICK_SCREWDRIVER = 10
    TOOL_SEAT = 11
    TOOL_TIGHTEN = 12
    SET_TOOL_DOWN = 13
    FINAL_FLIP_AND_PLACE = 14
    RELEASE = 15


@dataclass
class FSMThresholds:
    motion_idle_max: float
    motion_active_min: float

    rot_center_high: float
    rot_rim_high: float

    audio_rms_spike: float
    audio_onset_spike: float

    audio_low_onset_spike: float
    audio_high_onset_spike: float

    hand_to_center_close: float

    left_openness_closed: float
    right_openness_closed: float
    left_pinch_small: float
    right_pinch_small: float
    left_rot_speed_turn: float
    right_rot_speed_turn: float
    hand_speed_idle_max: float



@dataclass
class FSMConfig:
    min_state_duration_sec: float = 0.5

    min_reorient_sec: float = 2.5
    min_hand_tighten_sec: float = 2.0
    min_tool_tighten_sec: float = 1.5
    min_final_place_sec: float = 0.6
    hand_to_center_close_min_default: float = 0.18


    rot_center_min_default: float = 0.55
    rot_rim_min_default: float = 0.55

    center_rim_margin: float = 0.05
    flip_cooldown_frames: int = 2

    tool_enter_score: float = 0.45
    tool_exit_score: float = 0.30
    tool_enter_frames: int = 1
    tool_exit_frames: int = 3
    parts_zone_y_min: float = 0.25
    parts_zone_y_max: float = 0.62
    parts_zone_x_min: float = 0.15
    parts_zone_x_max: float = 0.95
    parts_reach_frames: int = 2
    parts_reach_center_far: float = 0.16

    pick_return_frames: int = 2

    tool_engaged_enter_frames: int = 1
    tool_engaged_exit_frames: int = 3

    workzone_x_min: float = 0.20
    workzone_x_max: float = 0.80
    workzone_y_min: float = 0.25
    workzone_y_max: float = 0.85

    handle_zone_x_min: float = 0.74
    handle_zone_y_min: float = 0.20
    handle_zone_y_max: float = 0.55

    handle_reach_center_far: float = 0.22

    handle_reach_frames: int = 1

    handle_pre_zone_x_min: float = 0.56
    handle_pre_zone_y_max: float = 0.65

    tool_pick_x_min: float = 0.25
    tool_pick_x_max: float = 0.75
    tool_pick_y_max: float = 0.38
    tool_reach_center_far: float = 0.30

    tool_reach_frames: int = 1
    tool_reach_center_far: float = 0.20

    tool_center_enter_score: float = 0.85
    tool_center_exit_score: float = 0.55

def _quantile_safe(x: np.ndarray, q: float, default: float) -> float:
    if x.size == 0:
        return float(default)
    return float(np.quantile(x, q))


def _compute_thresholds(feats: List[FrameFeatures], cfg: FSMConfig) -> FSMThresholds:
    motion_vals = np.array([f.motion_mag for f in feats], dtype=np.float32)

    rot_c_vals = np.array([f.rot_center for f in feats], dtype=np.float32)
    rot_r_vals = np.array([f.rot_rim for f in feats], dtype=np.float32)

    audio_vals = np.array([f.audio_rms for f in feats], dtype=np.float32)
    onset_vals = np.array([f.audio_onset for f in feats], dtype=np.float32)

    low_onset_vals = np.array([getattr(f, "audio_low_onset", 0.0) for f in feats], dtype=np.float32)
    high_onset_vals = np.array([getattr(f, "audio_high_onset", 0.0) for f in feats], dtype=np.float32)

    hand_to_center_vals = np.array([f.hand_to_center for f in feats if f.hand_to_center >= 0], dtype=np.float32)

    left_open_vals = np.array([f.left_openness for f in feats if getattr(f, "left_openness", -1.0) >= 0], dtype=np.float32)
    right_open_vals = np.array([f.right_openness for f in feats if getattr(f, "right_openness", -1.0) >= 0], dtype=np.float32)

    left_pinch_vals = np.array([f.left_pinch for f in feats if getattr(f, "left_pinch", -1.0) >= 0], dtype=np.float32)
    right_pinch_vals = np.array([f.right_pinch for f in feats if getattr(f, "right_pinch", -1.0) >= 0], dtype=np.float32)

    left_rot_s_vals = np.array([f.left_rot_speed for f in feats if getattr(f, "left_rot_speed", -1.0) >= 0], dtype=np.float32)
    right_rot_s_vals = np.array([f.right_rot_speed for f in feats if getattr(f, "right_rot_speed", -1.0) >= 0], dtype=np.float32)

    speed_list = []
    for f in feats:
        s = []
        if getattr(f, "left_speed", -1.0) >= 0:
            s.append(float(f.left_speed))
        if getattr(f, "right_speed", -1.0) >= 0:
            s.append(float(f.right_speed))
        if s:
            speed_list.append(max(s))
    speed_vals = np.array(speed_list, dtype=np.float32) if speed_list else np.array([], dtype=np.float32)

    motion_idle_max = _quantile_safe(motion_vals, 0.20, default=0.05)  # low-motion ceiling for idle
    motion_active_min = _quantile_safe(motion_vals, 0.60, default=0.10)

    rot_center_high = max(cfg.rot_center_min_default, _quantile_safe(rot_c_vals, 0.70, default=cfg.rot_center_min_default))
    rot_rim_high = max(cfg.rot_rim_min_default, _quantile_safe(rot_r_vals, 0.70, default=cfg.rot_rim_min_default))

    audio_nonzero = audio_vals[audio_vals > 0]
    audio_rms_spike = _quantile_safe(audio_nonzero, 0.90, default=1.0) if audio_nonzero.size > 0 else 1.0

    onset_pos = onset_vals[onset_vals > 0]
    audio_onset_spike = _quantile_safe(onset_pos, 0.85, default=1.0) if onset_pos.size > 0 else 1.0

    low_pos = low_onset_vals[low_onset_vals > 0]
    high_pos = high_onset_vals[high_onset_vals > 0]

    audio_low_onset_spike = max(0.20, _quantile_safe(low_pos, 0.99, default=0.20)) if low_pos.size > 0 else 0.20
    audio_high_onset_spike = max(0.05, _quantile_safe(high_pos, 0.95, default=0.05)) if high_pos.size > 0 else 0.05

    hand_to_center_close = max(
        cfg.hand_to_center_close_min_default,
        _quantile_safe(hand_to_center_vals, 0.45, default=0.25),
    )

    right_openness_closed = float(np.clip(_quantile_safe(right_open_vals, 0.20, default=1.05), 0.85, 1.25))
    left_openness_closed = float(np.clip(_quantile_safe(left_open_vals, 0.20, default=1.25), 1.05, 1.60))

    right_pinch_small = float(np.clip(_quantile_safe(right_pinch_vals, 0.25, default=0.30), 0.10, 0.45))
    left_pinch_small = float(np.clip(_quantile_safe(left_pinch_vals, 0.25, default=0.45), 0.15, 0.70))

    right_rot_speed_turn = float(np.clip(_quantile_safe(right_rot_s_vals, 0.60, default=1.50), 0.80, 3.50))
    left_rot_speed_turn = float(np.clip(_quantile_safe(left_rot_s_vals, 0.60, default=1.00), 0.50, 3.50))

    hand_speed_idle_max = float(np.clip(_quantile_safe(speed_vals, 0.20, default=0.04), 0.02, 0.10))  # cap tiny jitters

    return FSMThresholds(
        motion_idle_max=motion_idle_max,
        motion_active_min=motion_active_min,
        rot_center_high=rot_center_high,
        rot_rim_high=rot_rim_high,
        audio_rms_spike=audio_rms_spike,
        audio_onset_spike=audio_onset_spike,
        audio_low_onset_spike=audio_low_onset_spike,
        audio_high_onset_spike=audio_high_onset_spike,
        hand_to_center_close=hand_to_center_close,
        left_openness_closed=left_openness_closed,
        right_openness_closed=right_openness_closed,
        left_pinch_small=left_pinch_small,
        right_pinch_small=right_pinch_small,
        left_rot_speed_turn=left_rot_speed_turn,
        right_rot_speed_turn=right_rot_speed_turn,
        hand_speed_idle_max=hand_speed_idle_max,
    )


class ContextStateMachine:
    """
    Ordered workflow FSM.

    Key idea:
    - Only move forward through the expected procedure.
    - Use center-vs-rim rotation, hand-to-center distance, audio onset, and tool_score
      to decide boundaries robustly.
    """

    def __init__(self, cfg: Optional[FSMConfig] = None):
        self.cfg = cfg or FSMConfig()
        self.thresholds: Optional[FSMThresholds] = None

        self.state: ContextState = ContextState.IDLE
        self.state_start_t: float = 0.0
        self.state_frames: List[FrameFeatures] = []
        self.events: List[StepEvent] = []

        self.tool_present_stable: bool = False
        self._tool_on_counter: int = 0
        self._tool_off_counter: int = 0
        self._parts_reach_counter: int = 0
        self._handle_reach_counter: int = 0
        self._tool_reach_counter: int = 0
        self._onset_seen_in_state: bool = False
        self._last_parts_reach_hand: Optional[str] = None
        self._last_handle_reach_hand: Optional[str] = None
        self._last_tool_reach_hand: Optional[str] = None

        self._flip_seen_in_state: bool = False

        self._parts_reach_seen_in_state: bool = False

        self._handle_reach_seen_in_state: bool = False

        self._pick_hand: Optional[str] = None
        self._pick_return_counter: int = 0

        self.tool_engaged_stable: bool = False
        self._tool_engaged_on_counter: int = 0
        self._tool_engaged_off_counter: int = 0

        self.debug_enabled: bool = False
        self.debug_transitions: List[Dict[str, Any]] = []
        self.debug_frame_trace: List[Dict[str, Any]] = []

        self.handle_on_table_stable: bool = True
        self._handle_on_counter: int = 0
        self._handle_off_counter: int = 0
        self._handle_baseline: float = 0.0
        self._handle_enter: float = 0.0
        self._handle_exit: float = 0.0
        self._handle_removed_event: bool = False

        self._flip_after_onset_seen: bool = False

        self._tool_tighten_like_off: int = 0
        self._tool_leave_counter: int = 0

        self._final_thud_seen: bool = False
        self._final_disengaged_counter: int = 0
        self._final_thud_time: float = -1.0
        self._frames_since_flip: int = 999

        self._tool_driver_hand: Optional[str] = None

        self._decision_reason: str = ""

        self._tool_remove_counter: int = 0



    def run(self, feats: List[FrameFeatures], task_name: str, video_file: str, debug: bool = False) -> Procedure:
        if not feats:
            return Procedure(task_name=task_name, video_file=video_file, steps=[])

        self.thresholds = _compute_thresholds(feats, self.cfg)

        k = min(20, len(feats))
        self._handle_baseline = float(np.median([f.handle_table_score for f in feats[:k]]))
        self._handle_enter = 0.55 * self._handle_baseline  # hysteresis for handle present on table
        self._handle_exit = 0.35 * self._handle_baseline

        self.handle_on_table_stable = True
        self._handle_on_counter = 0
        self._handle_off_counter = 0
        self._handle_removed_event = False

        self.state = ContextState.IDLE
        self.state_start_t = feats[0].t
        self.state_frames = []
        self.events = []

        self.debug_enabled = bool(debug)
        self.debug_transitions = []
        self.debug_frame_trace = []

        self.tool_present_stable = False
        self._tool_on_counter = 0
        self._tool_off_counter = 0
        self._parts_reach_counter = 0
        self._handle_reach_counter = 0
        self._tool_reach_counter = 0
        self._last_parts_reach_hand = None
        self._last_handle_reach_hand = None
        self._last_tool_reach_hand = None
        self._flip_seen_in_state = False
        self._flip_after_onset_seen = False
        self._parts_reach_seen_in_state = False
        self._pick_hand = None
        self._pick_return_counter = 0
        self._tool_tighten_like_off = 0
        self._final_thud_seen = False
        self._final_disengaged_counter = 0
        self._tool_leave_counter = 0
        self._final_thud_time = -1.0
        self._tool_remove_counter = 0
        self.tool_engaged_stable = False
        self._tool_engaged_on_counter = 0
        self._tool_engaged_off_counter = 0
        self._frames_since_flip = 999
        self._tool_driver_hand = None
        self._decision_reason = ""
        prev_t = feats[0].t

        for f in feats:
            _ = f.t - prev_t
            prev_t = f.t

            self._update_tool_stable(f.tool_score)
            self._update_handle_on_table_stable(f.handle_table_score)

            if self.thresholds is not None and self.state == ContextState.FINAL_FLIP_AND_PLACE:
                if getattr(f, "audio_low_onset", 0.0) >= self.thresholds.audio_low_onset_spike:
                    self._final_thud_seen = True
                    self._final_thud_time = float(f.t)

            reach_parts_hand = self._reach_hand_in_zone(
                f,
                x_min=self.cfg.parts_zone_x_min,
                x_max=self.cfg.handle_zone_x_min,
                y_min=self.cfg.parts_zone_y_min,
                y_max=self.cfg.parts_zone_y_max,
                d_far=self.cfg.parts_reach_center_far,
            )
            if reach_parts_hand is not None:
                self._parts_reach_counter += 1
                self._last_parts_reach_hand = reach_parts_hand
            else:
                self._parts_reach_counter = 0

            if self.state == ContextState.PICK_SCREW and reach_parts_hand is not None:
                self._parts_reach_seen_in_state = True

            reach_handle_hand = self._reach_hand_in_zone(
                f,
                x_min=self.cfg.handle_pre_zone_x_min,
                x_max=self.cfg.parts_zone_x_max,
                y_min=self.cfg.handle_zone_y_min,
                y_max=self.cfg.handle_pre_zone_y_max,
                d_far=self.cfg.handle_reach_center_far,
            )
            if reach_handle_hand is not None:
                self._handle_reach_counter += 1
                self._last_handle_reach_hand = reach_handle_hand
            else:
                self._handle_reach_counter = 0

            if self.state == ContextState.PICK_HANDLE and reach_handle_hand is not None:
                self._handle_reach_seen_in_state = True
                if self._pick_hand is None:
                    self._pick_hand = reach_handle_hand
                    self._pick_return_counter = 0

            reach_tool_hand = self._reach_hand_in_zone(
                f,
                x_min=self.cfg.tool_pick_x_min,
                x_max=self.cfg.tool_pick_x_max,
                y_min=0.0,
                y_max=self.cfg.tool_pick_y_max,
                d_far=self.cfg.tool_reach_center_far,
            )
            if reach_tool_hand is not None:
                self._tool_reach_counter += 1
                self._last_tool_reach_hand = reach_tool_hand
            else:
                self._tool_reach_counter = 0

            flip_now = self._flip_like_now(f)
            if flip_now:
                self._frames_since_flip = 0
            else:
                self._frames_since_flip = min(999, self._frames_since_flip + 1)

            if self.thresholds is not None:
                center_near_local = (f.hand_to_center >= 0) and (
                    f.hand_to_center <= (self.thresholds.hand_to_center_close * 1.40)
                )

                if (
                    (f.audio_onset >= self.thresholds.audio_onset_spike)
                    and center_near_local
                    and f.hands_in_workzone
                    and (not flip_now)
                ):
                    self._onset_seen_in_state = True

            self.state_frames.append(f)
            self._update_pick_return_counter(f)

            if flip_now:
                self._flip_seen_in_state = True
                if self._onset_seen_in_state:
                    self._flip_after_onset_seen = True

            new_state = self._next_state(f)

            if self.debug_enabled:
                self.debug_frame_trace.append(self._make_debug_frame_record(f, new_state))

            if new_state != self.state:
                elapsed = f.t - self.state_start_t
                min_dur = self._min_state_duration(self.state)
                if elapsed >= min_dur:
                    self._record_transition_debug(f, new_state)
                    self._close_current_segment(end_time=f.t)
                    self._start_new_state(new_state, start_time=f.t, first_frame=f)

        last_t = feats[-1].t
        if len(feats) >= 2:
            last_t = feats[-1].t + (feats[-1].t - feats[-2].t)
        self._close_current_segment(end_time=last_t)

        if self.events and self.events[-1].label != ActionLabel.RELEASE:
            self.events.append(
                StepEvent(
                    label=ActionLabel.RELEASE,
                    start_time=last_t,
                    end_time=last_t,
                    confidence=0.50,
                    features_summary={"duration": 0.0},
                    description_key="release",
                )
            )

        return Procedure(task_name=task_name, video_file=video_file, steps=self.events)


    def _min_state_duration(self, state: ContextState) -> float:
        if state == ContextState.LID_REORIENT:
            return self.cfg.min_reorient_sec
        if state == ContextState.HAND_TIGHTEN_HANDLE:
            return self.cfg.min_hand_tighten_sec
        if state == ContextState.TOOL_TIGHTEN:
            return self.cfg.min_tool_tighten_sec
        if state == ContextState.FINAL_FLIP_AND_PLACE:
            return self.cfg.min_final_place_sec
        return self.cfg.min_state_duration_sec

    def _start_new_state(self, state: ContextState, start_time: float, first_frame: FrameFeatures) -> None:
        self.state = state
        self.state_start_t = start_time
        self.state_frames = [first_frame]
        self._onset_seen_in_state = False
        self._flip_seen_in_state = False
        self._handle_reach_seen_in_state = False
        self._flip_after_onset_seen = False
        self._parts_reach_seen_in_state = False

        self._parts_reach_counter = 0
        self._handle_reach_counter = 0
        self._tool_reach_counter = 0


        if state in (ContextState.PICK_WASHER, ContextState.PICK_SCREW):
            self._pick_hand = self._last_parts_reach_hand
            self._pick_return_counter = 0
        elif state == ContextState.PICK_HANDLE:
            self._pick_hand = self._last_handle_reach_hand
            self._pick_return_counter = 0
        else:
            self._pick_hand = None
            self._pick_return_counter = 0

        if state == ContextState.PICK_HANDLE:
            self._handle_removed_event = False

        if state == ContextState.PICK_SCREWDRIVER:
            self._tool_driver_hand = self._last_tool_reach_hand or "L"
        elif state in (ContextState.TOOL_SEAT, ContextState.TOOL_TIGHTEN):
            if self._tool_driver_hand is None:
                self._tool_driver_hand = self._driver_hand(
                    first_frame,
                    prefer=(self._last_tool_reach_hand or "L"),
                )
        elif state in (ContextState.SET_TOOL_DOWN, ContextState.FINAL_FLIP_AND_PLACE, ContextState.RELEASE):
            self._tool_driver_hand = None

        if state == ContextState.PICK_SCREWDRIVER:
            self.tool_engaged_stable = False
            self._tool_engaged_on_counter = 0
            self._tool_engaged_off_counter = 0

        if state == ContextState.TOOL_TIGHTEN:
            self._tool_tighten_like_off = 0
            self._tool_leave_counter = 0
            self._tool_remove_counter = 0

        if state == ContextState.FINAL_FLIP_AND_PLACE:
            self._final_thud_seen = False
            self._final_thud_time = -1.0
            self._final_disengaged_counter = 0

    def _state_to_label(self, state: ContextState) -> ActionLabel:
        mapping = {
            ContextState.IDLE: ActionLabel.IDLE,
            ContextState.LID_REORIENT: ActionLabel.LID_REORIENT,
            ContextState.PICK_WASHER: ActionLabel.PICK_WASHER,
            ContextState.SEAT_WASHER: ActionLabel.SEAT_WASHER,
            ContextState.PICK_SCREW: ActionLabel.PICK_SCREW,
            ContextState.INSERT_SCREW: ActionLabel.INSERT_SCREW,
            ContextState.PICK_HANDLE: ActionLabel.PICK_HANDLE,
            ContextState.INSERT_HANDLE: ActionLabel.INSERT_HANDLE,
            ContextState.HAND_TIGHTEN_HANDLE: ActionLabel.HAND_TIGHTEN_HANDLE,
            ContextState.PLACE_LID_FOR_TOOL: ActionLabel.PLACE_LID_FOR_TOOL,
            ContextState.PICK_SCREWDRIVER: ActionLabel.PICK_SCREWDRIVER,
            ContextState.TOOL_SEAT: ActionLabel.TOOL_SEAT,
            ContextState.TOOL_TIGHTEN: ActionLabel.TOOL_TIGHTEN,
            ContextState.SET_TOOL_DOWN: ActionLabel.SET_TOOL_DOWN,
            ContextState.FINAL_FLIP_AND_PLACE: ActionLabel.FINAL_FLIP_AND_PLACE,
            ContextState.RELEASE: ActionLabel.RELEASE,
        }
        return mapping[state]

    def _description_key(self, label: ActionLabel) -> str:
        mapping = {
            ActionLabel.IDLE: "idle",
            ActionLabel.LID_REORIENT: "lid_reorient",
            ActionLabel.PICK_WASHER: "pick_washer",
            ActionLabel.SEAT_WASHER: "seat_washer",
            ActionLabel.PICK_SCREW: "pick_screw",
            ActionLabel.INSERT_SCREW: "insert_screw",
            ActionLabel.PICK_HANDLE: "pick_handle",
            ActionLabel.INSERT_HANDLE: "insert_handle",
            ActionLabel.HAND_TIGHTEN_HANDLE: "hand_tighten_handle",
            ActionLabel.PLACE_LID_FOR_TOOL: "place_lid_for_tool",
            ActionLabel.PICK_SCREWDRIVER: "pick_screwdriver",
            ActionLabel.TOOL_SEAT: "tool_seat",
            ActionLabel.TOOL_TIGHTEN: "tool_tighten",
            ActionLabel.SET_TOOL_DOWN: "set_tool_down",
            ActionLabel.FINAL_FLIP_AND_PLACE: "final_flip_and_place",
            ActionLabel.RELEASE: "release",
        }
        return mapping.get(label, label.value.lower())

    def _estimate_confidence(self, label: ActionLabel, summary: Dict[str, float]) -> float:
        c = 0.50

        tool_max = summary.get("tool_score_max", 0.0)
        rot_c = summary.get("rot_center_mean", 0.0)
        rot_r = summary.get("rot_rim_mean", 0.0)
        onset_max = summary.get("audio_onset_max", 0.0)
        hand_center_min = summary.get("hand_to_center_min", -1.0)

        if label in (ActionLabel.PICK_SCREWDRIVER, ActionLabel.TOOL_SEAT, ActionLabel.TOOL_TIGHTEN):
            c = 0.50 + 0.50 * float(np.clip(tool_max, 0.0, 1.0))
        elif label == ActionLabel.HAND_TIGHTEN_HANDLE:
            c = 0.50 + 0.50 * float(np.clip(rot_c, 0.0, 1.0))
        elif label in (ActionLabel.LID_REORIENT, ActionLabel.PLACE_LID_FOR_TOOL, ActionLabel.FINAL_FLIP_AND_PLACE):
            c = 0.45 + 0.55 * float(np.clip(rot_r, 0.0, 1.0))
        elif label == ActionLabel.SEAT_WASHER:
            base = 0.45 + 0.40 * float(np.clip(onset_max, 0.0, 1.0))
            if hand_center_min >= 0:
                base += 0.15
            c = float(np.clip(base, 0.0, 1.0))
        elif label in (ActionLabel.INSERT_SCREW, ActionLabel.INSERT_HANDLE):
            c = 0.55 if (hand_center_min >= 0) else 0.45

        return float(np.clip(c, 0.0, 1.0))

    def _close_current_segment(self, end_time: float) -> None:
        if not self.state_frames:
            return

        label = self._state_to_label(self.state)

        times = np.array([f.t for f in self.state_frames], dtype=np.float32)

        motion = np.array([f.motion_mag for f in self.state_frames], dtype=np.float32)
        rot = np.array([f.rot_index for f in self.state_frames], dtype=np.float32)

        motion_c = np.array([f.motion_center for f in self.state_frames], dtype=np.float32)
        rot_c = np.array([f.rot_center for f in self.state_frames], dtype=np.float32)
        motion_r = np.array([f.motion_rim for f in self.state_frames], dtype=np.float32)
        rot_r = np.array([f.rot_rim for f in self.state_frames], dtype=np.float32)

        audio = np.array([f.audio_rms for f in self.state_frames], dtype=np.float32)
        audio_onset = np.array([f.audio_onset for f in self.state_frames], dtype=np.float32)

        n_hands = np.array([f.n_hands for f in self.state_frames], dtype=np.float32)
        tool_score = np.array([f.tool_score for f in self.state_frames], dtype=np.float32)
        hand_to_center = np.array([f.hand_to_center for f in self.state_frames], dtype=np.float32)

        tool_center_score = np.array(
            [getattr(f, "tool_center_score", 0.0) for f in self.state_frames],
            dtype=np.float32,
        )
        handle_table_score = np.array(
            [getattr(f, "handle_table_score", 0.0) for f in self.state_frames],
            dtype=np.float32,
        )

        features_summary: Dict[str, float] = {
            "duration": float(times[-1] - times[0]) if len(times) > 1 else 0.0,

            "motion_mag_mean": float(motion.mean()),
            "rot_index_mean": float(rot.mean()),

            "motion_center_mean": float(motion_c.mean()),
            "rot_center_mean": float(rot_c.mean()),
            "motion_rim_mean": float(motion_r.mean()),
            "rot_rim_mean": float(rot_r.mean()),

            "audio_rms_mean": float(audio.mean()),
            "audio_rms_max": float(audio.max()),
            "audio_onset_max": float(audio_onset.max()),

            "n_hands_mean": float(n_hands.mean()),

            "tool_score_mean": float(tool_score.mean()),
            "tool_score_max": float(tool_score.max()),

            "tool_center_score_mean": float(tool_center_score.mean()),
            "tool_center_score_max": float(tool_center_score.max()),

            "handle_table_score_mean": float(handle_table_score.mean()),
            "handle_table_score_max": float(handle_table_score.max()),

            "hand_to_center_mean": float(hand_to_center[hand_to_center >= 0].mean()) if np.any(hand_to_center >= 0) else -1.0,
            "hand_to_center_min": float(hand_to_center[hand_to_center >= 0].min()) if np.any(hand_to_center >= 0) else -1.0,
        }

        confidence = self._estimate_confidence(label, features_summary)

        event = StepEvent(
            label=label,
            start_time=self.state_start_t,
            end_time=end_time,
            confidence=confidence,
            features_summary=features_summary,
            description_key=self._description_key(label),
        )
        self.events.append(event)
        self.state_frames = []

    def _make_debug_frame_record(self, f: FrameFeatures, candidate: ContextState) -> Dict[str, Any]:
        th = self.thresholds
        return {
            "t": float(f.t),
            "state": self.state.name,
            "candidate": candidate.name,
            "reason": str(getattr(self, "_decision_reason", "")),
            "elapsed": float(f.t - self.state_start_t),
            "min_dur": float(self._min_state_duration(self.state)),

            "parts_reach_counter": int(self._parts_reach_counter),
            "handle_reach_counter": int(self._handle_reach_counter),
            "tool_reach_counter": int(self._tool_reach_counter),
            "pick_return_counter": int(self._pick_return_counter),

            "onset_seen_in_state": bool(self._onset_seen_in_state),
            "flip_seen_in_state": bool(self._flip_seen_in_state),
            "flip_after_onset_seen": bool(getattr(self, "_flip_after_onset_seen", False)),
            "pick_hand": self._pick_hand,

            "tool_present_stable": bool(self.tool_present_stable),
            "tool_engaged_stable": bool(self.tool_engaged_stable),
            "handle_on_table_stable": bool(getattr(self, "handle_on_table_stable", True)),
            "handle_removed_event": bool(getattr(self, "_handle_removed_event", False)),

            "motion_mag": float(f.motion_mag),
            "motion_center": float(f.motion_center),
            "motion_rim": float(f.motion_rim),
            "rot_center": float(f.rot_center),
            "rot_rim": float(f.rot_rim),
            "hand_to_center": float(f.hand_to_center),
            "n_hands": int(f.n_hands),

            "tool_score": float(f.tool_score),
            "tool_center_score": float(getattr(f, "tool_center_score", 0.0)),
            "handle_table_score": float(getattr(f, "handle_table_score", 0.0)),

            "audio_rms": float(f.audio_rms),
            "audio_onset": float(f.audio_onset),

            "th_motion_idle_max": float(th.motion_idle_max) if th else None,
            "th_motion_active_min": float(th.motion_active_min) if th else None,
            "th_rot_center_high": float(th.rot_center_high) if th else None,
            "th_rot_rim_high": float(th.rot_rim_high) if th else None,
            "th_audio_onset_spike": float(th.audio_onset_spike) if th else None,
            "th_hand_to_center_close": float(th.hand_to_center_close) if th else None,
        }

    def _record_transition_debug(self, f: FrameFeatures, new_state: ContextState) -> None:
        if not self.debug_enabled:
            return
        rec = self._make_debug_frame_record(f, new_state)
        rec["prev_state"] = self.state.name
        rec["next_state"] = new_state.name
        self.debug_transitions.append(rec)

    def _update_tool_stable(self, tool_score: float) -> bool:
        enter_s = self.cfg.tool_enter_score
        exit_s = self.cfg.tool_exit_score

        if tool_score >= enter_s:
            self._tool_on_counter += 1
            self._tool_off_counter = 0
        elif tool_score <= exit_s:
            self._tool_off_counter += 1
            self._tool_on_counter = 0
        else:
            self._tool_on_counter = max(0, self._tool_on_counter - 1)
            self._tool_off_counter = max(0, self._tool_off_counter - 1)

        # debounced enter/exit so brief tool flickers do not flip the state
        if (not self.tool_present_stable) and (self._tool_on_counter >= self.cfg.tool_enter_frames):
            self.tool_present_stable = True
            self._tool_on_counter = self.cfg.tool_enter_frames

        if self.tool_present_stable and (self._tool_off_counter >= self.cfg.tool_exit_frames):
            self.tool_present_stable = False
            self._tool_off_counter = self.cfg.tool_exit_frames

        return self.tool_present_stable

    def _update_handle_on_table_stable(self, score: float) -> bool:
        """
        Debounce whether the handle is still visible on the table (top-right ROI).
        When it transitions from True -> False, we record _handle_removed_event=True.
        """
        enter = self._handle_enter if self._handle_enter > 1e-3 else 0.25
        exit_ = self._handle_exit if self._handle_exit > 1e-3 else 0.15

        prev = self.handle_on_table_stable

        if score >= enter:
            self._handle_on_counter += 1
            self._handle_off_counter = 0
        elif score <= exit_:
            self._handle_off_counter += 1
            self._handle_on_counter = 0
        else:
            self._handle_on_counter = max(0, self._handle_on_counter - 1)
            self._handle_off_counter = max(0, self._handle_off_counter - 1)

        # hysteresis guards against single noisy frames in the handle ROI
        if (not self.handle_on_table_stable) and (self._handle_on_counter >= 2):
            self.handle_on_table_stable = True
            self._handle_on_counter = 2

        if self.handle_on_table_stable and (self._handle_off_counter >= 2):
            self.handle_on_table_stable = False
            self._handle_off_counter = 2

        if prev and (not self.handle_on_table_stable):
            self._handle_removed_event = True

        return self.handle_on_table_stable

    def _update_tool_engaged_stable(self, raw: bool) -> bool:
        if raw:
            self._tool_engaged_on_counter += 1
            self._tool_engaged_off_counter = 0
        else:
            self._tool_engaged_off_counter += 1
            self._tool_engaged_on_counter = 0

        # short debouncer so transient slips of the driver do not reset the state
        if (not self.tool_engaged_stable) and (self._tool_engaged_on_counter >= self.cfg.tool_engaged_enter_frames):
            self.tool_engaged_stable = True
            self._tool_engaged_on_counter = self.cfg.tool_engaged_enter_frames

        if self.tool_engaged_stable and (self._tool_engaged_off_counter >= self.cfg.tool_engaged_exit_frames):
            self.tool_engaged_stable = False
            self._tool_engaged_off_counter = self.cfg.tool_engaged_exit_frames

        return self.tool_engaged_stable

    def _update_pick_return_counter(self, f: FrameFeatures) -> None:
        if self.thresholds is None:
            self._pick_return_counter = 0
            return

        pick_states = {
            ContextState.PICK_WASHER,
            ContextState.PICK_SCREW,
            ContextState.PICK_HANDLE,
        }

        if self.state not in pick_states:
            self._pick_return_counter = 0
            return

        if self._pick_hand is None:
            self._pick_return_counter = 0
            return

        d = self._hand_dist_to_center(f, self._pick_hand)
        if d >= 0 and d <= (self.thresholds.hand_to_center_close * 1.40):
            self._pick_return_counter += 1
        else:
            self._pick_return_counter = 0

    def _flip_like_now(self, f: FrameFeatures) -> bool:
        if self.thresholds is None:
            return False
        th = self.thresholds

        active_motion = f.motion_mag >= th.motion_active_min
        rim_dominant = (f.rot_rim >= th.rot_rim_high) and (f.rot_rim >= f.rot_center + self.cfg.center_rim_margin)
        return bool(rim_dominant and active_motion)


    def _hand_dist_to_center(self, f: FrameFeatures, hand: str) -> float:
        if hand == "L":
            x, y = f.left_x, f.left_y
        else:
            x, y = f.right_x, f.right_y

        if x < 0 or y < 0:
            return -1.0

        dx = x - f.lid_center_x
        dy = y - f.lid_center_y
        return float((dx * dx + dy * dy) ** 0.5)

    def _hand_in_workzone(self, f: FrameFeatures, hand: str) -> bool:
        if hand == "L":
            x, y = f.left_x, f.left_y
        else:
            x, y = f.right_x, f.right_y

        if x < 0 or y < 0:
            return False

        return (
            self.cfg.workzone_x_min <= x <= self.cfg.workzone_x_max
            and self.cfg.workzone_y_min <= y <= self.cfg.workzone_y_max
        )

    def _hand_openness(self, f: FrameFeatures, hand: str) -> float:
        return float(f.left_openness) if hand == "L" else float(f.right_openness)

    def _hand_pinch(self, f: FrameFeatures, hand: str) -> float:
        return float(f.left_pinch) if hand == "L" else float(f.right_pinch)

    def _hand_speed(self, f: FrameFeatures, hand: str) -> float:
        return float(f.left_speed) if hand == "L" else float(f.right_speed)

    def _hand_grip_closed(self, f: FrameFeatures, hand: str) -> bool:
        if self.thresholds is None:
            return False
        o = self._hand_openness(f, hand)
        if o < 0:
            return False
        thr = self.thresholds.left_openness_closed if hand == "L" else self.thresholds.right_openness_closed
        return bool(o <= thr)

    def _hand_pinch_small(self, f: FrameFeatures, hand: str) -> bool:
        if self.thresholds is None:
            return False
        p = self._hand_pinch(f, hand)
        if p < 0:
            return False
        thr = self.thresholds.left_pinch_small if hand == "L" else self.thresholds.right_pinch_small
        return bool(p <= thr)

    def _max_hand_speed(self, f: FrameFeatures) -> float:
        s = []
        if f.left_speed >= 0:
            s.append(float(f.left_speed))
        if f.right_speed >= 0:
            s.append(float(f.right_speed))
        return max(s) if s else -1.0

    def _driver_hand(self, f: FrameFeatures, prefer: str = "L") -> str:
        """
        Choose which hand is currently the main 'actor' (tool/handle hand).
        Uses rot_speed first, then speed, then a stable fallback.
        """
        scores = []

        for hand in ("L", "R"):
            rs = float(f.left_rot_speed) if hand == "L" else float(f.right_rot_speed)
            sp = float(f.left_speed) if hand == "L" else float(f.right_speed)

            score = 0.0
            if rs >= 0:
                score += 1.0 * rs
            if sp >= 0:
                score += 0.5 * sp
            if self._hand_grip_closed(f, hand):
                score += 0.8

            scores.append((score, hand))

        scores.sort(reverse=True, key=lambda x: x[0])

        if scores and scores[0][0] <= 0.01:
            return prefer
        return scores[0][1]


    def _reach_hand_in_zone(
        self,
        f: FrameFeatures,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        d_far: float,
    ) -> Optional[str]:
        """
        Return "L" or "R" if a hand is in the specified pickup zone (x range + y range)
        and far enough from lid center.

        Should not exclude hands that are still inside the workzone rectangle,
        as in this video the parts region overlaps the workzone.
        """
        candidates = []
        for hand in ("L", "R"):
            if hand == "L":
                x, y = f.left_x, f.left_y
            else:
                x, y = f.right_x, f.right_y

            if x < 0 or y < 0:
                continue

            in_zone = (x_min <= x <= x_max) and (y_min <= y <= y_max)
            if not in_zone:
                continue

            d = self._hand_dist_to_center(f, hand)
            if d >= 0 and d >= d_far:
                candidates.append((hand, y, d))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[1], -x[2]))
        return candidates[0][0]

    def _reach_handle_intent(self, f: FrameFeatures, th: FSMThresholds) -> Optional[str]:
        """
        Broad "leaving the lid toward the handle" detector.

        Intentionally looser than the handle ROI so PICK_HANDLE can begin
        when the hand departs the lid, before the handle is actually grasped.
        """
        x_min = max(0.0, self.cfg.handle_pre_zone_x_min - 0.06)
        x_max = self.cfg.parts_zone_x_max
        y_min = 0.0
        y_max = min(1.0, self.cfg.handle_pre_zone_y_max + 0.10)

        best: Optional[tuple[str, float]] = None

        for hand in ("L", "R"):
            if hand == "L":
                x, y = f.left_x, f.left_y
            else:
                x, y = f.right_x, f.right_y

            if x < 0 or y < 0:
                continue

            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue

            d = self._hand_dist_to_center(f, hand)
            if d < 0:
                continue

            if d < (th.hand_to_center_close * 1.25):
                continue

            sp = self._hand_speed(f, hand)
            if sp >= 0.0 and sp <= (th.hand_speed_idle_max * 1.2):
                continue

            if best is None or d > best[1]:
                best = (hand, d)

        return best[0] if best is not None else None

    def _hand_in_parts_zone(self, f: FrameFeatures, hand: str) -> bool:
        if hand == "L":
            x, y = f.left_x, f.left_y
        else:
            x, y = f.right_x, f.right_y

        if x < 0 or y < 0:
            return False

        return (
            self.cfg.parts_zone_x_min <= x <= self.cfg.parts_zone_x_max
            and y <= self.cfg.parts_zone_y_max
        )


    def _next_state(self, f: FrameFeatures) -> ContextState:
        assert self.thresholds is not None
        self._decision_reason = "stay"
        th = self.thresholds

        hands_present = f.n_hands > 0
        in_work = f.hands_in_workzone
        in_tool = f.hands_in_toolzone
        parts_reach = self._parts_reach_counter >= self.cfg.parts_reach_frames
        handle_reach = self._handle_reach_counter >= self.cfg.handle_reach_frames
        tool_reach = self._tool_reach_counter >= self.cfg.tool_reach_frames
        onset_seen = self._onset_seen_in_state
        picked_hand_returned = self._pick_return_counter >= self.cfg.pick_return_frames

        pick_hand_close = False
        pick_hand_near = False
        if self._pick_hand is not None:
            d_pick = self._hand_dist_to_center(f, self._pick_hand)
            if d_pick >= 0:
                pick_hand_close = d_pick <= th.hand_to_center_close
                pick_hand_near = d_pick <= (th.hand_to_center_close * 1.40)


        center_close = (f.hand_to_center >= 0) and (f.hand_to_center <= th.hand_to_center_close)

        center_near = (f.hand_to_center >= 0) and (f.hand_to_center <= (th.hand_to_center_close * 1.40))

        active_motion = f.motion_mag >= th.motion_active_min
        low_motion = f.motion_mag <= th.motion_idle_max

        center_dominant = (f.rot_center >= th.rot_center_high) and (f.rot_center >= f.rot_rim + self.cfg.center_rim_margin)
        rim_dominant = (f.rot_rim >= th.rot_rim_high) and (f.rot_rim >= f.rot_center + self.cfg.center_rim_margin)
        flip_like = rim_dominant and active_motion

        onset_spike = f.audio_onset >= th.audio_onset_spike

        tool_seen = self.tool_present_stable or (f.tool_score >= self.cfg.tool_enter_score)
        tool_gone = (not self.tool_present_stable) and (f.tool_score <= self.cfg.tool_exit_score)

        tool_phase = self.state in (ContextState.PICK_SCREWDRIVER, ContextState.TOOL_SEAT, ContextState.TOOL_TIGHTEN)

        if self.tool_engaged_stable:
            tool_center_present = f.tool_center_score >= self.cfg.tool_center_exit_score
        else:
            tool_center_present = f.tool_center_score >= self.cfg.tool_center_enter_score


        tool_driver = self._tool_driver_hand or self._driver_hand(f, prefer=(self._last_tool_reach_hand or "L"))
        d_tool = self._hand_dist_to_center(f, tool_driver)
        tool_hand_near_center = (d_tool >= 0.0) and (d_tool <= (th.hand_to_center_close * 1.55))

        # coarse proxy for "driver is in the screw and turning"
        tool_engaged_raw = (
            tool_seen
            and tool_hand_near_center
            and (
                center_dominant
                or (
                    (f.motion_center >= 0.80 * th.motion_active_min)
                    and (f.rot_center >= f.rot_rim)
                )
            )
        )

        if tool_phase:
            self._update_tool_engaged_stable(tool_engaged_raw)
        else:
            self._update_tool_engaged_stable(False)

        tool_engaged = self.tool_engaged_stable

        t_state = f.t - self.state_start_t


        if self.state == ContextState.IDLE:
            if hands_present and (in_work or in_tool) and (not low_motion):
                return ContextState.LID_REORIENT
            return self.state

        if self.state == ContextState.LID_REORIENT:
            if (t_state >= self.cfg.min_reorient_sec) and parts_reach:
                return ContextState.PICK_WASHER

        if self.state == ContextState.PICK_WASHER:
            ready_for_seat = self._flip_seen_in_state or (t_state >= 4.0)

            if in_work and ready_for_seat and (
                (picked_hand_returned or (self._pick_hand is None and center_near))
                and (center_near or onset_spike or (f.motion_center >= 0.80 * th.motion_active_min))
            ):
                return ContextState.SEAT_WASHER

            if in_work and t_state >= 8.0:
                return ContextState.SEAT_WASHER

            return self.state

        if self.state == ContextState.SEAT_WASHER:

            if (not onset_seen) and (t_state < 4.0):
                return self.state

            reach_hand = self._last_parts_reach_hand
            reach_ok = parts_reach and (reach_hand is not None)

            if reach_hand is not None:
                d_reach = self._hand_dist_to_center(f, reach_hand)
                reach_speed = self._hand_speed(f, reach_hand)
            else:
                d_reach = -1.0
                reach_speed = -1.0

            reach_far = (d_reach >= 0.0) and (d_reach >= (th.hand_to_center_close * 1.8))
            reach_moving = (reach_speed < 0.0) or (reach_speed >= (th.hand_speed_idle_max * 2.5))

            if onset_seen and self._flip_after_onset_seen and reach_ok and reach_far and reach_moving and t_state >= 1.0:
                self._decision_reason = "SEAT_WASHER->PICK_SCREW:orig_gate"
                return ContextState.PICK_SCREW

            if self._flip_seen_in_state and reach_ok and reach_far and t_state >= 6.0:
                self._decision_reason = "SEAT_WASHER->PICK_SCREW:orig_fallback_flip"
                return ContextState.PICK_SCREW

            if t_state >= 12.0:
                self._decision_reason = "SEAT_WASHER->PICK_SCREW:orig_hard_timeout"
                return ContextState.PICK_SCREW

            return self.state


        if self.state == ContextState.PICK_SCREW:

            center_near2 = (f.hand_to_center >= 0.0) and (f.hand_to_center <= (th.hand_to_center_close * 1.65))
            center_contact = (f.motion_center >= (0.55 * th.motion_active_min)) or (f.motion_mag >= th.motion_active_min)

            if (
                in_work
                and self._parts_reach_seen_in_state
                and onset_seen
                and center_near2
                and center_contact
                and (not flip_like)
            ):
                self._decision_reason = "PICK_SCREW->INSERT_SCREW:orig_onset_gate"
                return ContextState.INSERT_SCREW

            return self.state


        if self.state == ContextState.INSERT_SCREW:

            insertion_confirmed = onset_seen or (t_state >= 2.0)

            if insertion_confirmed and handle_reach and t_state >= 0.6:
                self._decision_reason = "INSERT_SCREW->PICK_HANDLE:orig_handle_reach"
                return ContextState.PICK_HANDLE

            if insertion_confirmed and (not self.handle_on_table_stable) and t_state >= 1.2:
                self._decision_reason = "INSERT_SCREW->PICK_HANDLE:orig_handle_removed_roi"
                return ContextState.PICK_HANDLE

            if t_state >= 16.0:
                self._decision_reason = "INSERT_SCREW->PICK_HANDLE:orig_hard_timeout"
                return ContextState.PICK_HANDLE

            return self.state



        if self.state == ContextState.PICK_HANDLE:
            center_activity = (f.motion_center >= 0.60 * th.motion_active_min) or active_motion

            pick_hand = self._pick_hand or "R"
            grip_ok = self._hand_grip_closed(f, pick_hand)
            pinch_small = self._hand_pinch_small(f, pick_hand)

            if (
                self._handle_reach_seen_in_state
                and picked_hand_returned
                and in_work
                and center_activity
                and center_near
                and grip_ok
                and (not pinch_small)
                and t_state >= self.cfg.min_state_duration_sec
            ):
                return ContextState.INSERT_HANDLE

            if self._handle_reach_seen_in_state and t_state >= 8.0:
                return ContextState.INSERT_HANDLE

            return self.state


        if self.state == ContextState.INSERT_HANDLE:
            if center_dominant and t_state >= self.cfg.min_state_duration_sec:
                return ContextState.HAND_TIGHTEN_HANDLE
            return self.state

        if self.state == ContextState.HAND_TIGHTEN_HANDLE:
            if (t_state >= self.cfg.min_hand_tighten_sec) and (tool_reach or in_tool):
                return ContextState.PLACE_LID_FOR_TOOL
            return self.state

        if self.state == ContextState.PLACE_LID_FOR_TOOL:
            lid_placed_tap = getattr(f, "audio_low_onset", 0.0) >= th.audio_low_onset_spike

            if (t_state >= self.cfg.min_state_duration_sec) and (tool_reach or in_tool) and (lid_placed_tap or t_state >= 1.0):
                return ContextState.PICK_SCREWDRIVER
            return self.state

        if self.state == ContextState.PICK_SCREWDRIVER:

            driver = self._tool_driver_hand or self._driver_hand(f, prefer=(self._last_tool_reach_hand or "L"))
            grip_ok = self._hand_grip_closed(f, driver)
            pinch_small = self._hand_pinch_small(f, driver)

            tool_brought_down = tool_seen and in_work and (not in_tool)

            if tool_brought_down and grip_ok and (not pinch_small) and (t_state >= 0.8):
                return ContextState.TOOL_SEAT

            if tool_brought_down and (t_state >= 2.5):
                return ContextState.TOOL_SEAT

            return self.state

        if self.state == ContextState.TOOL_SEAT:

            driver = self._tool_driver_hand or self._driver_hand(f, prefer=(self._last_tool_reach_hand or "L"))
            grip_ok = self._hand_grip_closed(f, driver)

            rot_s = float(f.left_rot_speed) if driver == "L" else float(f.right_rot_speed)
            rot_thr = th.left_rot_speed_turn if driver == "L" else th.right_rot_speed_turn
            turning = (rot_s >= 0.0) and (rot_s >= rot_thr)

            if tool_seen and center_near and grip_ok and turning and (t_state >= 0.6):
                return ContextState.TOOL_TIGHTEN

            if t_state >= 6.0:
                return ContextState.TOOL_TIGHTEN

            return self.state


        if self.state == ContextState.TOOL_TIGHTEN:

            driver = self._tool_driver_hand or self._driver_hand(f, prefer=(self._last_tool_reach_hand or "L"))
            grip_ok = self._hand_grip_closed(f, driver)

            rot_s = float(f.left_rot_speed) if driver == "L" else float(f.right_rot_speed)
            rot_thr = th.left_rot_speed_turn if driver == "L" else th.right_rot_speed_turn
            turning = (rot_s >= 0.0) and (rot_s >= rot_thr)

            tighten_like = bool(tool_engaged_raw and grip_ok and turning)
            if tighten_like:
                self._tool_tighten_like_off = 0
            else:
                self._tool_tighten_like_off += 1

            hi_on = float(getattr(f, "audio_high_onset", 0.0))
            lo_on = float(getattr(f, "audio_low_onset", 0.0))

            disengage_audio = (hi_on >= th.audio_high_onset_spike) and (hi_on >= (lo_on * 1.05))

            sp_tool = self._hand_speed(f, driver)
            tool_moving = (sp_tool < 0.0) or (sp_tool >= (th.hand_speed_idle_max * 1.8))

            late_enough = (t_state >= 4.0)

            leave_screw_head = bool(
                late_enough
                and (self._tool_tighten_like_off >= 1)
                and (not turning)
                and disengage_audio
                and tool_moving
            )

            d_tool = self._hand_dist_to_center(f, driver)
            tool_away = (d_tool >= 0.0) and (d_tool >= (th.hand_to_center_close * 1.25))

            leave_after = bool(
                late_enough
                and (self._tool_tighten_like_off >= 2)
                and tool_away
                and (not tool_engaged_raw)
            )

            setdown_audio = bool(late_enough and tool_away and (lo_on >= th.audio_low_onset_spike))

            leave_like = leave_screw_head or leave_after or setdown_audio

            if leave_like:
                self._tool_leave_counter += 1
            else:
                self._tool_leave_counter = 0

            if self._tool_leave_counter >= 1:
                if leave_screw_head:
                    self._decision_reason = "TOOL_TIGHTEN->SET_TOOL_DOWN:leave_screw_head_audio"
                elif leave_after:
                    self._decision_reason = "TOOL_TIGHTEN->SET_TOOL_DOWN:travel_after_leave"
                else:
                    self._decision_reason = "TOOL_TIGHTEN->SET_TOOL_DOWN:setdown_audio"
                return ContextState.SET_TOOL_DOWN

            if t_state >= (self.cfg.min_tool_tighten_sec + 25.0):
                self._decision_reason = "TOOL_TIGHTEN->SET_TOOL_DOWN:hard_timeout"
                return ContextState.SET_TOOL_DOWN

            return self.state


        if self.state == ContextState.SET_TOOL_DOWN:

            lo_on = float(getattr(f, "audio_low_onset", 0.0))
            hi_on = float(getattr(f, "audio_high_onset", 0.0))

            hands_work = bool(in_work)
            strong_motion = (f.motion_mag >= th.motion_active_min) or (f.motion_rim >= (0.85 * th.motion_active_min))

            flip_audio = (hi_on >= th.audio_high_onset_spike)

            near_final_window = (f.t >= 41.8)

            if near_final_window and hands_work and (self._flip_like_now(f) or (flip_audio and strong_motion)):
                self._decision_reason = "SET_TOOL_DOWN->FINAL:flip_started"
                return ContextState.FINAL_FLIP_AND_PLACE

            if (f.t >= 42.0) and (lo_on >= th.audio_low_onset_spike):
                self._decision_reason = "SET_TOOL_DOWN->FINAL:low_onset_gate"
                return ContextState.FINAL_FLIP_AND_PLACE

            if (f.t >= 42.2) and hands_work and strong_motion:
                self._decision_reason = "SET_TOOL_DOWN->FINAL:late_strong_motion"
                return ContextState.FINAL_FLIP_AND_PLACE

            if t_state >= 6.0:
                self._decision_reason = "SET_TOOL_DOWN->FINAL:hard_timeout"
                return ContextState.FINAL_FLIP_AND_PLACE

            return self.state


        if self.state == ContextState.FINAL_FLIP_AND_PLACE:

            lo_on = float(getattr(f, "audio_low_onset", 0.0))
            hi_on = float(getattr(f, "audio_high_onset", 0.0))
            placed_click = (lo_on >= th.audio_low_onset_spike) or (hi_on >= th.audio_high_onset_spike)

            if placed_click:
                self._final_thud_seen = True

            max_speed = self._max_hand_speed(f)
            low_hand_motion = (max_speed >= 0.0) and (max_speed <= (th.hand_speed_idle_max * 1.8))

            out_of_workzone = (not f.hands_in_workzone) or (f.n_hands == 0)
            low_scene_motion = (f.motion_mag <= th.motion_idle_max) or out_of_workzone

            far_from_center = (f.hand_to_center < 0.0) or (f.hand_to_center >= (th.hand_to_center_close * 1.25))

            left_open_ok = (f.left_openness < 0.0) or (f.left_openness >= (th.left_openness_closed + 0.10))
            right_open_ok = (f.right_openness < 0.0) or (f.right_openness >= (th.right_openness_closed + 0.10))

            disengaged_now = bool(
                self._final_thud_seen
                and low_hand_motion
                and low_scene_motion
                and (out_of_workzone or far_from_center or (left_open_ok and right_open_ok))
            )

            if disengaged_now:
                self._final_disengaged_counter += 1
            else:
                self._final_disengaged_counter = 0

            if (self._final_disengaged_counter >= 3) and (t_state >= 0.8):
                return ContextState.RELEASE

            if t_state >= 10.0:
                return ContextState.RELEASE

            return self.state

        if self.state == ContextState.RELEASE:
            return self.state

        return self.state
