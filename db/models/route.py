"""db/models/route.py"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from uuid import UUID, uuid4

BoardSource = Literal["kilter", "moonboard", "custom"]
HoldRole = Literal["start", "hand", "foot", "finish"]


@dataclass
class BoardRoute:
    source: BoardSource
    board_type: str
    board_angle_deg: int
    id: UUID = field(default_factory=uuid4)
    external_id: Optional[str] = None
    name: Optional[str] = None
    setter_name: Optional[str] = None
    community_grade: Optional[str] = None
    difficulty_score: Optional[float] = None   # normalised 0.0–1.0
    send_count: int = 0
    avg_quality_rating: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    synced_at: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not (0 <= self.board_angle_deg <= 70):
            raise ValueError(f"board_angle_deg must be 0–70, got {self.board_angle_deg}")
        if self.difficulty_score is not None and not (0.0 <= self.difficulty_score <= 1.0):
            raise ValueError(f"difficulty_score must be 0.0–1.0, got {self.difficulty_score}")


@dataclass
class RouteHold:
    route_id: UUID
    grid_col: int
    grid_row: int
    role: HoldRole
    id: UUID = field(default_factory=uuid4)
    position_x_cm: Optional[float] = None
    position_y_cm: Optional[float] = None
    hold_type: Optional[str] = None   # 'crimp', 'sloper', 'jug', etc.
    board_angle_deg: Optional[float] = None
    hand_sequence: Optional[int] = None   # ordering among hand holds

    def distance_to(self, other: RouteHold) -> Optional[float]:
        """Euclidean distance in cm to another hold. Requires position_x/y_cm."""
        if None in (self.position_x_cm, self.position_y_cm,
                    other.position_x_cm, other.position_y_cm):
            return None
        return (
            (self.position_x_cm - other.position_x_cm) ** 2
            + (self.position_y_cm - other.position_y_cm) ** 2
        ) ** 0.5
