"""db/models/session.py and move.py"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from uuid import UUID, uuid4

VideoStatus = Literal["none", "pending", "processing", "done", "failed"]
HandUsed = Literal["left", "right", "both", "unknown"]


# ── Sessions & Attempts ───────────────────────────────

@dataclass
class Session:
    climber_id: UUID
    gym_id: UUID
    board_angle_deg: int
    board_type: str
    id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    video_path: Optional[str] = None
    video_status: VideoStatus = "none"
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def duration_minutes(self) -> Optional[float]:
        if self.ended_at and self.started_at:
            return (self.ended_at - self.started_at).seconds / 60
        return None


@dataclass
class Attempt:
    session_id: UUID
    route_id: UUID
    climber_id: UUID
    id: UUID = field(default_factory=uuid4)
    attempt_number: int = 1
    completed: bool = False
    duration_sec: Optional[float] = None
    video_offset_sec: Optional[float] = None
    video_end_sec: Optional[float] = None
    attempted_at: datetime = field(default_factory=datetime.utcnow)


# ── Move Events & Pose ────────────────────────────────

@dataclass
class MoveEvent:
    attempt_id: UUID
    to_hold_id: UUID
    sequence_order: int
    id: UUID = field(default_factory=uuid4)
    from_hold_id: Optional[UUID] = None   # NULL on first move
    move_type: Optional[str] = None       # set by ML classifier
    reach_distance_cm: Optional[float] = None
    lateral_distance_cm: Optional[float] = None
    vertical_distance_cm: Optional[float] = None
    hand_used: HandUsed = "unknown"
    timestamp_sec: Optional[float] = None
    succeeded: Optional[bool] = None


@dataclass
class PoseFrame:
    attempt_id: UUID
    timestamp_sec: float
    raw_landmarks: dict          # full MediaPipe 33-point output
    id: UUID = field(default_factory=uuid4)
    move_event_id: Optional[UUID] = None
    hip_angle_deg: Optional[float] = None
    shoulder_rot_deg: Optional[float] = None
    left_arm_reach_norm: Optional[float] = None
    right_arm_reach_norm: Optional[float] = None
    left_foot_angle_deg: Optional[float] = None
    right_foot_angle_deg: Optional[float] = None
    com_height_norm: Optional[float] = None
    tension_score: Optional[float] = None   # 0.0–1.0, computed


@dataclass
class MoveLibrary:
    name: str           # 'crimp', 'heel_hook', 'dyno', etc.
    category: str       # 'grip_strength', 'body_position', 'dynamic', 'footwork', 'technique'
    id: UUID = field(default_factory=uuid4)
    description: Optional[str] = None
    required_body_positions: list[str] = field(default_factory=list)
    primary_muscles: list[str] = field(default_factory=list)
    difficulty_tier: Optional[str] = None
    grade_range_low: Optional[str] = None
    grade_range_high: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ClimberMoveStats:
    climber_id: UUID
    move_library_id: UUID
    id: UUID = field(default_factory=uuid4)
    total_attempts: int = 0
    successful_attempts: int = 0
    vs_peers_percentile: Optional[int] = None   # 0–100
    exposure_ratio: Optional[float] = None       # <1.0 = under-exposed gap
    is_flagged_weakness: bool = False
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> Optional[float]:
        if self.total_attempts == 0:
            return None
        return self.successful_attempts / self.total_attempts
