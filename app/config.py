import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment variables and constants
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://<your-id>.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "<your-anon-key>")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "overwatch-replays")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "matches")
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", os.path.join("app", "models", "best.pt"))

KNOWN_MODES = [
    "COMPETITIVE ROLE QUEUE",
    "COMPETITIVE OPEN QUEUE",
    "OVERWATCH: CLASSIC",
    "UNRANKED",
    "QUICK PLAY",
    "ARCADE",
    "CUSTOM GAME",
    "MYSTERY HEROES",
    "NO LIMITS",
    "TOTAL MAYHEM",
]

KNOWN_RESULTS = ["VICTORY!", "DEFEAT!", "DRAW!"]

KNOWN_MAPS = [
    # Control
    "Antarctic Peninsula",
    "Busan",
    "Ilios",
    "Lijiang Tower",
    "Nepal",
    "Oasis",
    "Samoa",
    # Escort
    "Circuit Royal",
    "Dorado",
    "Havana",
    "Junkertown",
    "Rialto",
    "Route 66",
    "Shambali Monastery",
    "Watchpoint: Gibraltar",
    # Flashpoint
    "New Junk City",
    "Suravasa",
    # Hybrid
    "Blizzard World",
    "Eichenwalde",
    "Hollywood",
    "King's Row",
    "Midtown",
    "Numbani",
    "Paraíso",
    # Push
    "Colosseo",
    "Esperança",
    "New Queen Street",
    "Runasapi",
    # Clash
    "Hanaoka",
    "Throne of Anubis",
]
