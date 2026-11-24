"""Shared typed containers passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# 17 COCO-style joints (all pose engines) + 4 foot points (heel, toe) that MediaPipe also provides. The foot points make
# foot-contact timing and left/right leg identity more reliable; they are NaN for engines that don't output them.
COCO_JOINTS: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
FOOT_JOINTS: list[str] = ["left_heel", "right_heel", "left_foot_index", "right_foot_index"]
ALL_JOINTS: list[str] = COCO_JOINTS + FOOT_JOINTS
N_JOINTS = len(ALL_JOINTS)
MP_INDEX: list[int] = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
J = {name: i for i, name in enumerate(ALL_JOINTS)}
LEG_PARTS = ("knee", "ankle", "heel", "foot_index")  # swapped together when left/right leg labels are corrected

SKELETON_EDGES: list[tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"), ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"), ("right_ankle", "right_foot_index"),
]

ProgressFn = Callable[[float], None]


@dataclass
class VideoMeta:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float
    rotation: int = 0
    codec: str = ""
    vfr: bool = False
    process_fps: Optional[float] = None  # rate the clip is converted to (VFR phone clips: the nominal rate, see quality.py)


@dataclass
class PoseSequence:
    """Pose of the selected person. kp: (F, N_JOINTS, 3) = x_px, y_px (image coords), visibility; NaN = not detected.
    z: (F, N_JOINTS) relative depth (smaller = nearer the camera) when the engine provides it, else None.
    raw: every detection in every frame, kept so the bowler can be re-selected (another candidate) if needed."""
    fps: float
    width: int
    height: int
    kp: np.ndarray
    device: str
    people_detected_max: int = 1
    detection_rate: float = 0.0
    active_span: Optional[tuple[int, int]] = None  # frames where the bowler is on screen (inclusive)
    z: Optional[np.ndarray] = None
    raw: Optional[object] = None
    selected_by: str = "auto"  # "tap" | "auto" | "auto-retry"
    tiled: bool = False  # found by the enlarged-quarters pass (bowler too small for the full-frame detector)
    zoomed: Optional[tuple[int, float]] = None  # (bowler height px, zoom factor) when refined on a zoomed crop
    cam: Optional[object] = None  # adapt.CameraPath when the camera moved: maps image coords to a fixed reference view
    tids: Optional[list] = None  # ByteTrack identity of the bowler per frame (RTMDet engine)
    head_neck: Optional[np.ndarray] = None  # (F, 2, 3) head-top and neck points (Halpe-26), for 3D lifting
    secondary: Optional[np.ndarray] = None  # (F, N_JOINTS, 3) the secondary estimator's keypoints (MediaPipe)
    disagreement: Optional[np.ndarray] = None  # (F, N_JOINTS) primary vs secondary distance, body heights
    engine: str = ""  # estimator that produced kp
    fusion: Optional[dict] = None  # summary of the primary/secondary comparison


@dataclass
class BowlerSelection:
    """User guidance for picking the bowler: a tap on the bowler (frame, x_px, y_px) and areas to ignore (px boxes)."""
    tap: Optional[tuple[int, float, float]] = None
    ignore: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class Kinematics:
    """Smoothed kinematics in the analysis frame: mirrored so bowling is +x, Y up, pixels.

    xy: (F, N_JOINTS, 2); vis: (F, N_JOINTS); valid: (F, N_JOINTS) bool (measured or short-gap interpolated).
    img_xy: (F, N_JOINTS, 2) smoothed coordinates in original image space (x right, y down), for drawing.
    """
    fps: float
    width: int
    height: int
    xy: np.ndarray
    vis: np.ndarray
    valid: np.ndarray
    img_xy: np.ndarray
    bowling_arm: str  # "right" | "left"
    direction: str  # "left_to_right" | "right_to_left"
    interpolated_frac: float = 0.0
    roles: Optional[dict] = None  # identity.assign_roles: {"front_leg", "back_leg", ...}; None until key moments exist
    identity_log: list = field(default_factory=list)  # swap corrections made while smoothing
    head_neck: Optional[np.ndarray] = None  # (F, 2, 2) head-top and neck (image coords) when the engine gives them

    @property
    def n(self) -> int:
        return int(self.xy.shape[0])

    def joint(self, name: str) -> np.ndarray:
        return self.xy[:, J[name], :]

    def side(self, part: str, which: str) -> str:
        """which in {'bowling','front','back','non_bowling'} → anatomical joint name."""
        arm = self.bowling_arm
        other = "left" if arm == "right" else "right"
        if self.roles and which in ("front", "back"):  # roles from the foot contacts, not the raw labels
            return f"{self.roles[which + '_leg']}_{part}"
        side = {"bowling": arm, "non_bowling": other, "back": arm, "front": other}[which]
        return f"{side}_{part}"

    def mid(self, a: str, b: str) -> np.ndarray:
        return 0.5 * (self.joint(a) + self.joint(b))

    def joint_vis(self, name: str) -> np.ndarray:
        return self.vis[:, J[name]]


@dataclass
class EventResult:
    name: str
    frame: int
    confidence: str
    reason: str
    sharpness: float
    methods: dict = field(default_factory=dict)  # each detection method's frame (None = not found)
    agree: int = 0  # how many methods agreed (consensus.py)
    score: float = 0.0  # consensus confidence score 0–1


@dataclass
class DeliveryEvents:
    bfc: EventResult
    ffc: EventResult
    release: EventResult
    method: str = "contact_first"  # "release_first" when found from a validated release peak (more trustworthy)
    legs_swapped: bool = False  # the front foot (by position) carries the "back" leg label: relabel the legs
    needs_review: bool = False  # methods disagreed or a moment was estimated: ask the user to check
    review_reasons: list = field(default_factory=list)


@dataclass
class Scale:
    body_height_px: float
    frame: int
    metres_per_pixel: Optional[float]
    method: str  # "height" | "body_relative"
    rel_uncertainty: float = 0.0


@dataclass
class QualityReport:
    level: str
    reasons: list[str]
    fps: float
    width: int
    height: int
    duration_s: float
    rotation: int
    camera_motion: Optional[float]
    view_detected: str
    checks: list[dict] = field(default_factory=list)
