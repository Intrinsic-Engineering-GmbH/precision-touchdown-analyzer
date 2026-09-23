"""The points a landing earns from its offset. Stdlib only.

The rules are the club's, not the software's: the full score on the target
line, the width of that line (it is painted, not infinitely thin, so there
is a band around it that still earns everything), a deduction per metre
short and a different one per metre long (a short landing is usually
punished harder), a floor, and what a landing outside the measurement
window gets. They live in ``config/scoring.json`` and are edited from the
Scoring page.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CONFIG_NAME = "scoring.json"


@dataclass(slots=True)
class ScoringRules:
    max_points: float = 100.0
    target_width_m: float = 0.0  # full width of the line; max points within half of it
    short_per_m: float = 5.0  # points lost per metre before the line
    long_per_m: float = 2.0  # points lost per metre beyond the line
    min_points: float = 0.0  # the floor for a measured landing
    out_of_range_points: float = 0.0  # a landing outside the window (< / > bound)
    decimals: int = 0  # how the score is rounded
    name: str = "Club rules"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def target_half_width_m(self) -> float:
        """How far either side of the line still earns everything."""
        return max(0.0, self.target_width_m) / 2.0

    def miss_m(self, longitudinal_m: float) -> float:
        """Distance deducted for: the offset less the half width, never below 0.

        Touching down anywhere on the painted line is a bullseye, so the
        deduction is measured from the edge of the line, not its centre.
        """
        return max(0.0, abs(longitudinal_m) - self.target_half_width_m)

    def score(self, longitudinal_m: float | None, outcome: str) -> float | None:
        """Points for one landing, or ``None`` when it is not scored at all.

        ``longitudinal_m`` is signed: negative is short of the line.
        Anything that is not a landing (take-off, fly-through, rolling)
        earns nothing; a landing measured outside the window, or bounded
        beyond it, gets the out-of-range points.
        """
        if outcome in ("short", "long"):
            return round(self.out_of_range_points, self.decimals)
        if outcome != "measured" or longitudinal_m is None:
            return None
        rate = self.short_per_m if longitudinal_m < 0 else self.long_per_m
        points = self.max_points - rate * self.miss_m(longitudinal_m)
        return round(max(self.min_points, points), self.decimals)


def load(config_dir: Path) -> ScoringRules:
    path = config_dir / CONFIG_NAME
    if not path.is_file():
        return ScoringRules()
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        known = set(ScoringRules.__dataclass_fields__)
        return ScoringRules(**{k: v for k, v in payload.items() if k in known})
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("could not read %s: %s", path, exc)
        return ScoringRules()


def save(config_dir: Path, rules: ScoringRules) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / CONFIG_NAME
    path.write_text(json.dumps(rules.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path
