"""Bounded, source-clock scan cadence; this module never commands a vehicle.

A fast scan remains a real detector inference, not a promise of real-time speed.
Late detail windows are discarded instead of being queued for catch-up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class ScanPlan:
    timestamp: float
    profile: str
    generation: int
    reset: bool
    detail_index: int
    fast_count: int
    next_due: float
    lateness: float
    skipped: int
    deferred: bool
    source_advanced: bool
    previous_detail: float | None
    _owner: object = field(repr=False, compare=False)


class HybridScanScheduler:
    """First scan fast, then at least N fast scans between periodic details.

    ``plan`` is read-only; commit only after a successful inference. For replay,
    supply source capture time, not inference wall time. Seeking backwards starts
    a fresh cadence without carrying old deadlines across the seek.
    """

    def __init__(self, interval_seconds=5., full_frame_every=3, min_fast_frames=2):
        if (type(interval_seconds) not in (int, float) or not math.isfinite(interval_seconds)
                or not .5 <= interval_seconds <= 3600):
            raise ValueError("Detail interval must be finite in 0.5..3600 seconds")
        if type(full_frame_every) is not int or not 1 <= full_frame_every <= 100:
            raise ValueError("Full detail cadence must be an integer in 1..100")
        if type(min_fast_frames) is not int or not 2 <= min_fast_frames <= 1000:
            raise ValueError("Use at least 2 and at most 1000 fast frames between detail sweeps")
        self.interval_seconds = float(interval_seconds)
        self.full_frame_every = full_frame_every
        self.min_fast_frames = min_fast_frames
        self._owner = object()
        self._generation = 0
        self._last_timestamp = None
        self._last_detail = None
        self._next_due = None
        self._detail_index = 0
        self._fast_count = 0
        self._skipped = 0

    def plan(self, timestamp_seconds):
        if (type(timestamp_seconds) not in (int, float) or not math.isfinite(timestamp_seconds)
                or not 0 <= timestamp_seconds <= 1e12):
            raise ValueError("Scan timestamp must be finite in 0..1e12 seconds")
        timestamp = float(timestamp_seconds)
        reset = self._last_timestamp is not None and timestamp < self._last_timestamp
        fresh = reset or self._last_timestamp is None
        deadline = timestamp + self.interval_seconds if fresh else self._next_due
        detail_index = 0 if fresh else self._detail_index
        fast_count = 0 if fresh else self._fast_count
        previous_detail = None if fresh else self._last_detail
        source_advanced = fresh or timestamp > self._last_timestamp
        due = timestamp >= deadline
        lateness = max(0., timestamp - deadline)
        detail = due and source_advanced and fast_count >= self.min_fast_frames
        skipped = 0
        if detail:
            detail_index += 1
            profile = "full-tiles" if detail_index % self.full_frame_every == 0 else "ground-tiles"
            fast_count = 0
            skipped = int(lateness // self.interval_seconds)
            deadline = timestamp + self.interval_seconds
        else:
            profile = "fast"
            if source_advanced:
                fast_count += 1
        return ScanPlan(timestamp, profile, self._generation, reset, detail_index,
                        fast_count, deadline, lateness, skipped, due and not detail and source_advanced,
                        source_advanced, previous_detail, self._owner)

    def commit(self, plan, seconds):
        if not isinstance(plan, ScanPlan) or plan._owner is not self._owner or plan.generation != self._generation:
            raise ValueError("Scan plan is stale or belongs to another scheduler")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Scan duration must be finite and nonnegative")
        self._generation += 1
        self._last_timestamp = plan.timestamp
        self._next_due = plan.next_due
        self._fast_count = plan.fast_count
        self._detail_index = plan.detail_index
        self._skipped = (0 if plan.reset else self._skipped) + plan.skipped
        detail = plan.profile != "fast"
        self._last_detail = plan.timestamp if detail else plan.previous_detail
        return {
            "schema": "mycoscout.hybrid-scan-cadence.v1",
            "requestedIntervalSeconds": self.interval_seconds,
            "minimumFastFramesBetweenDetails": self.min_fast_frames,
            "fullFrameEveryDetailSweeps": self.full_frame_every,
            "sourceTimestampSeconds": plan.timestamp,
            "nextDetailDueTimestampSeconds": plan.next_due,
            "detailSweepIndex": plan.detail_index,
            "fastFramesSinceDetail": plan.fast_count,
            "detailOverdueBySeconds": plan.lateness,
            "skippedDetailWindows": plan.skipped,
            "totalSkippedDetailWindows": self._skipped,
            "detailDeferredForFastFrames": plan.deferred,
            "detailDeferredForUnchangedTimestamp": not plan.source_advanced and plan.timestamp >= plan.next_due,
            "observedDetailIntervalSeconds": plan.timestamp - plan.previous_detail
                if detail and plan.previous_detail is not None else None,
            "clockReset": plan.reset,
            "sourceTimestampAdvanced": plan.source_advanced,
            "scanSeconds": float(seconds),
            "exceedsRequestedInterval": seconds > self.interval_seconds,
            "synchronousInference": True,
            "queuedDetailScans": 0,
            "scope": "whole-frame-context" if not detail else
                "whole-frame-detail" if plan.profile == "full-tiles" else "lower-image-detail",
            "scopeMeaning": "Image regions, not semantic terrain or species identification",
            "actuationEnabled": False,
        }
