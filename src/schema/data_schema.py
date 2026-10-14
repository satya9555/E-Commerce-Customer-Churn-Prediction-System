from pydantic import BaseModel
from typing import List, Optional


class TimeFilterConfig(BaseModel):
    enabled: bool = False
    date_column: str = "Date"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class DataConfig(BaseModel):
    input_path: str
    processed_path: str
    required_columns: List[str]
    target_column: str
    allowed_missing_pct: float = 0.2
    time_filter: TimeFilterConfig = TimeFilterConfig()
