import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Redis Configuration (Upstash)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")  # Empty if no password

    # Rate Limiting Configuration
    RATE_LIMITS: List[str] = os.getenv("RATE_LIMITS", "5/minute;25/hour;100/day")
