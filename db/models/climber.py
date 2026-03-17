"""db/models/gym.py and climber.py"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class Gym:
    name: str
    timezone: str = "America/Chicago"
    id: UUID = field(default_factory=uuid4)
    location: Optional[str] = None
    kilter_board_angle: Optional[int] = None   # degrees 0–70
    has_moon_board: bool = False
    has_kilter_board: bool = False
    contact_email: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.kilter_board_angle is not None:
            if not (0 <= self.kilter_board_angle <= 70):
                raise ValueError(f"kilter_board_angle must be 0–70, got {self.kilter_board_angle}")


@dataclass
class Climber:
    gym_id: UUID
    name: str
    id: UUID = field(default_factory=uuid4)
    email: Optional[str] = None
    self_reported_grade: Optional[str] = None   # e.g. 'V5', '7a'
    goals: list[str] = field(default_factory=list)
    weak_move_types: list[str] = field(default_factory=list)
    years_climbing: Optional[int] = None
    height_cm: Optional[int] = None
    wingspan_cm: Optional[int] = None
    video_consent: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def ape_index(self) -> Optional[float]:
        """Wingspan minus height in cm. Positive = long arms."""
        if self.height_cm and self.wingspan_cm:
            return self.wingspan_cm - self.height_cm
        return None
