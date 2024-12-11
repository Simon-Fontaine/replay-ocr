import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Rate Limiting Configuration
    RATE_LIMITS: List[str] = os.getenv("RATE_LIMITS", "5/minute;25/hour;100/day")
