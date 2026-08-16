"""JSON and CSV export helpers for a completed video-intelligence session."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict

from .engine import VideoIntelligenceEngine


def build_report(engine: VideoIntelligenceEngine, video_duration_seconds: float) -> Dict[str, Any]:
    report = engine.summary()
    report["video_duration_seconds"] = round(video_duration_seconds, 2)
    report["timeline"] = [event.to_row() for event in engine.events.events]
    report["highest_risk_event"] = (
        {
            "track_id": engine.highest_risk.track_id,
            "risk_score": engine.highest_risk.risk_score,
            "anomaly_score": engine.highest_risk.anomaly.score,
            "reason": engine.highest_risk.risk_message,
        }
        if engine.highest_risk
        else None
    )
    return report


def report_json(report: Dict[str, Any]) -> bytes:
    return json.dumps(report, indent=2).encode("utf-8")


def timeline_csv(engine: VideoIntelligenceEngine) -> bytes:
    rows = [event.to_row() for event in engine.events.events]
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue().encode("utf-8")
