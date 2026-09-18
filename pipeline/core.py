"""
The interchangeable-module contract (article Section 4.7 / dissertation
position 2): every risk-modeling method consumes a `RunContext` built over
the unified spatio-temporal-species model and returns a `RiskResult` in one
of a few standard shapes. Visualization and reporting code depends only on
`RiskResult`, never on which module produced it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


@dataclass
class RunContext:
    disease: str
    events: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskResult:
    geojson: dict | None
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]


class RiskModule(ABC):
    name: str
    input_kind: Literal["point_events", "farm_survey"]
    produces: Literal["cluster_table", "suitability_surface", "posterior", "components"]

    @abstractmethod
    def validate(self, ctx: RunContext) -> list[str]:
        """Return non-fatal warnings; raise ValueError for fatal input problems."""

    @abstractmethod
    def run(self, ctx: RunContext) -> RiskResult:
        ...
