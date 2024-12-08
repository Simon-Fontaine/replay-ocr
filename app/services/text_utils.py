import re
from datetime import datetime, timedelta
from app.config import KNOWN_RESULTS, KNOWN_MODES, KNOWN_MAPS


def format_match_result(result_text: str):
    if not result_text:
        return None, None

    parts = result_text.split("|")
    if len(parts) != 2:
        # If no pipe, try last token as score
        tokens = result_text.split()
        if len(tokens) > 1 and re.match(r"\d+-\d+", tokens[-1]):
            result = " ".join(tokens[:-1])
            score = tokens[-1]
            return validate_result(result), score
        return None, None

    result = parts[0].strip()
    score = parts[1].strip()
    return validate_result(result), score


def validate_result(result: str) -> str:
    if result in KNOWN_RESULTS:
        return result
    upper_res = result.upper()
    if "VICT" in upper_res:
        return "VICTORY!"
    elif "DEFE" in upper_res:
        return "DEFEAT!"
    elif "DRAW" in upper_res:
        return "DRAW!"
    return result


def sanitize_replay_code(code: str) -> str:
    code = code.upper()
    return re.sub(r"[^A-Z0-9]", "", code)


def fix_time_format(time_str: str) -> str:
    time_str = re.sub(r"[.;]", ":", time_str.strip())
    if ":" not in time_str and len(time_str) == 4 and time_str.isdigit():
        hh, mm = time_str[:2], time_str[2:]
        if 0 <= int(hh) < 24 and 0 <= int(mm) < 60:
            time_str = f"{hh}:{mm}"
    return time_str


def best_mode_match(mode: str) -> str:
    if not mode:
        return "COMPETITIVE ROLE QUEUE"
    for m in KNOWN_MODES:
        if m.upper() == mode.upper():
            return m

    upper_mode = mode.upper()
    if "COMPETITIVE" in upper_mode and "ROLE" in upper_mode:
        return "COMPETITIVE ROLE QUEUE"
    elif "OPEN" in upper_mode and "QUEUE" in upper_mode:
        return "COMPETITIVE OPEN QUEUE"
    elif "UNRANKED" in upper_mode:
        return "UNRANKED"
    elif "CLASSIC" in upper_mode:
        return "OVERWATCH: CLASSIC"
    elif "QUICK" in upper_mode and "PLAY" in upper_mode:
        return "QUICK PLAY"
    elif "ARCADE" in upper_mode:
        return "ARCADE"
    elif "CUSTOM" in upper_mode and "GAME" in upper_mode:
        return "CUSTOM GAME"

    return "COMPETITIVE ROLE QUEUE"


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    v0 = list(range(len(b) + 1))
    v1 = [0] * (len(b) + 1)
    for i in range(len(a)):
        v1[0] = i + 1
        for j in range(len(b)):
            cost = 0 if a[i] == b[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0
    return v0[len(b)]


def best_map_match(text: str) -> str:
    if not text:
        return ""
    upper_text = text.upper()
    candidates = [(m, levenshtein(upper_text, m.upper())) for m in KNOWN_MAPS]
    candidates.sort(key=lambda x: x[1])
    best_map, dist = candidates[0]
    if dist < len(text) / 2:
        return best_map
    return text


def parse_duration_to_datetime(time_ago: str, duration: str) -> str:
    days_ago = 0
    match_days = re.search(r"(\d+)\s+DAY", duration.upper())
    if match_days:
        days_ago = int(match_days.group(1))

    match_time = re.match(r"(\d{1,2}):(\d{2})", time_ago)
    if match_time:
        hours = int(match_time.group(1))
        minutes = int(match_time.group(2))
    else:
        return time_ago

    now = datetime.now()
    final_time = (now - timedelta(days=days_ago)).replace(
        hour=hours, minute=minutes, second=0, microsecond=0
    )
    return final_time.isoformat()
