
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class ActionLabel(str, Enum):
    IDLE = "IDLE"

    LID_REORIENT = "LID_REORIENT"
    PICK_WASHER = "PICK_WASHER"
    SEAT_WASHER = "SEAT_WASHER"
    PICK_SCREW = "PICK_SCREW"
    INSERT_SCREW = "INSERT_SCREW"
    PICK_HANDLE = "PICK_HANDLE"
    INSERT_HANDLE = "INSERT_HANDLE"
    HAND_TIGHTEN_HANDLE = "HAND_TIGHTEN_HANDLE"

    PLACE_LID_FOR_TOOL = "PLACE_LID_FOR_TOOL"
    PICK_SCREWDRIVER = "PICK_SCREWDRIVER"
    TOOL_SEAT = "TOOL_SEAT"
    TOOL_TIGHTEN = "TOOL_TIGHTEN"
    SET_TOOL_DOWN = "SET_TOOL_DOWN"

    FINAL_FLIP_AND_PLACE = "FINAL_FLIP_AND_PLACE"
    RELEASE = "RELEASE"



@dataclass
class FrameFeatures:
    """
    Features extracted for a single analysis frame.

    All coordinates are in normalized image space [0, 1],
    where (0, 0) is top left and (1, 1) is bottom right.
    Missing values use -1.0 as a sentinel.
    """

    t: float
    lid_center_x: float
    lid_center_y: float

    motion_mag: float
    rot_index: float

    motion_center: float
    rot_center: float
    motion_rim: float
    rot_rim: float

    n_hands: int
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    hand_dist: float
    hand_to_center: float
    hands_in_workzone: bool
    hands_in_toolzone: bool

    left_speed: float
    right_speed: float

    left_openness: float
    right_openness: float

    left_pinch: float
    right_pinch: float

    left_rot_speed: float
    right_rot_speed: float

    tool_score: float
    tool_present: bool

    audio_rms: float
    audio_onset: float

    audio_low_rms: float
    audio_high_rms: float
    audio_low_onset: float
    audio_high_onset: float

    handle_table_score: float

    tool_center_score: float

@dataclass
class StepEvent:
    """
    One contiguous action segment between two state changes of the FSM.
    """

    label: ActionLabel
    start_time: float
    end_time: float
    confidence: float
    features_summary: Dict[str, float]
    description_key: str


@dataclass
class Procedure:
    """
    High level representation of a whole task execution.
    """

    task_name: str
    video_file: str
    steps: List[StepEvent]
