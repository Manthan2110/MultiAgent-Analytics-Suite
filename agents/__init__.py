from dataclasses import dataclass, field
import pandas as pd
from typing import Dict, Any, List


@dataclass
class AgentContext:
    dataset_path: str
    df: pd.DataFrame | None = None
    summary: Dict[str, Any] = field(default_factory=dict)
    plots: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    report_path: str | None = None