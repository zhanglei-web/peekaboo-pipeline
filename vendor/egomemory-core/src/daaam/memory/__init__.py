"""Persistent editable and cross-session map memory."""

from .committed import CommittedEgoMemory
from .lifelong import LifelongConfig, LifelongMemory, LifelongObservation, replay_map_memory

from .store import (
    CorrectionReceipt,
    KnowledgeReceipt,
    KnowledgeRecordReceipt,
    MapMemory,
    MapMemoryConfig,
    PositionChangeReceipt,
    SessionRegistration,
)
from .semantic_worker import DeliveredSemanticCorrection, VersionedCorrectionProcessor
from .temporal import (
    DetectedPositionChange,
    PositionChangeConfig,
    TrajectoryPoint,
    detect_position_changes,
)

__all__ = [
    "CommittedEgoMemory",
    "LifelongConfig",
    "LifelongMemory",
    "LifelongObservation",
    "replay_map_memory",
    "CorrectionReceipt",
    "KnowledgeReceipt",
    "KnowledgeRecordReceipt",
    "DetectedPositionChange",
    "DeliveredSemanticCorrection",
    "MapMemory",
    "MapMemoryConfig",
    "PositionChangeConfig",
    "PositionChangeReceipt",
    "SessionRegistration",
    "TrajectoryPoint",
    "VersionedCorrectionProcessor",
    "detect_position_changes",
]
