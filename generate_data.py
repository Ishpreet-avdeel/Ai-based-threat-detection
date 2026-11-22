"""
generate_data.py
----------------
Generates a synthetic system-log dataset (data/system_logs.csv) for testing and
demoing the AI Threat Detection System.

The CSV contains exactly 5 columns (one event per row):
    timestamp, user_id, ip_address, action, status

Contents:
    * ~1,000 rows of normal working-hours traffic
    * Attack 1 - after-hours logins from an external IP   (EMP_105)
    * Attack 2 - brute-force login burst, then a success  (EMP_108)
    * Attack 3 - abnormal requests-per-minute spike       (EMP_112)

Usage:
    python generate_data.py                                   # default: ~1,000 rows
    python generate_data.py --rows 50000 --days 14 --users 50 # large dataset
    python generate_data.py --rows 200000 --days 30 --users 100 --output data/big.csv
"""

import argparse
import random
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Faker is listed in requirements.txt and is used for random choices below.
# We fall back to the standard library so the script never hard-crashes.
try:
    from faker import Faker
except ImportError:  # pragma: no cover
    Faker = None

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "system_logs.csv"

COLUMNS = ["timestamp", "user_id", "ip_address", "action", "status"]
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_USERS = 20     # EMP_101 ... EMP_120
DEFAULT_ROWS = 1000
DEFAULT_DAYS = 3
MIN_USERS = 12         # attacks use EMP_105, EMP_108 and EMP_112
MAX_USERS = 200        # keeps internal IPs inside 192.168.1.x
ATTACK_DAY = datetime(2026, 11, 20)  # last day of the dataset
MINUTES_PER_DAY = 8 * 60             # 09:00-16:59

# Relative likelihood of each action in normal traffic (must sum to 1.0)
ACTION_WEIGHTS = OrderedDict(
    [
        ("LOGIN", 0.10),
        ("VIEW_FILE", 0.40),
        ("DOWNLOAD", 0.15),
        ("API_CALL", 0.25),
        ("LOGOUT", 0.10),
    ]
)
SUCCESS_RATE = 0.98

# Filled in by configure() once the number of users is known
USERS: list = []
USER_IPS: dict = {}
DAYS: list = []

# Attack parameters
EXTERNAL_IP = "45.33.22.11"
SPIKE_USER = "EMP_112"
SPIKE_MINUTE = ATTACK_DAY + timedelta(hours=11, minutes=30)  # 11:30:00


def configure(n_users: int, n_days: int) -> None:
    """Build the user list, per-user internal IPs and the list of days."""
    USERS[:] = [f"EMP_{100 + n}" for n in range(1, n_users + 1)]
    USER_IPS.clear()
    # Each employee always connects from the same internal address (192.168.1.x)
    USER_IPS.update({user: f"192.168.1.{10 + i}" for i, user in enumerate(USERS)})
    DAYS[:] = [ATTACK_DAY - timedelta(days=d) for d in range(n_days - 1, -1, -1)]


# --------------------------------------------------------------------------- #
# Random helpers (Faker-backed, stdlib fallback)
# --------------------------------------------------------------------------- #
if Faker is not None:
    Faker.seed(SEED)
    _fake = Faker()
else:
    _fake = None


def pick_action() -> str:
    """Pick a normal-traffic action according to ACTION_WEIGHTS."""
    if _fake is not None:
        return _fake.random_element(elements=ACTION_WEIGHTS)
    return random.choices(list(ACTION_WEIGHTS), weights=list(ACTION_WEIGHTS.values()))[0]


def pick_status() -> str:
    """98% SUCCESS / 2% FAILED."""
    return "SUCCESS" if random.random() < SUCCESS_RATE else "FAILED"


def make_row(ts: datetime, user: str, ip: str, action: str, status: str) -> dict:
    """Build one log row using the exact CSV schema."""
    return {
        "timestamp": ts.strftime(TIMESTAMP_FORMAT),
        "user_id": user,
        "ip_address": ip,
        "action": action,
        "status": status,
    }


# --------------------------------------------------------------------------- #
# Normal traffic
# --------------------------------------------------------------------------- #
def generate_normal_traffic(target_rows: int = DEFAULT_ROWS) -> list:
    """
    Generate ~target_rows of normal activity between 09:00:00 and 16:59:59.

    Each (user, minute) bucket is used at most once and holds 1-4 events, so
    normal activity never exceeds 4 requests per minute per user.
    """
    rows, used_buckets = [], set()

    while len(rows) < target_rows:
        day = random.choice(DAYS)
        user = random.choice(USERS)
        minute_start = day + timedelta(hours=9, minutes=random.randint(0, MINUTES_PER_DAY - 1))

        bucket = (user, minute_start)
        # Skip used buckets and keep the spike minute clean for the attack
        if bucket in used_buckets or bucket == (SPIKE_USER, SPIKE_MINUTE):
            continue
        used_buckets.add(bucket)

        n_events = min(random.randint(1, 4), target_rows - len(rows))
        for sec in sorted(random.sample(range(60), n_events)):
            rows.append(
                make_row(
                    minute_start + timedelta(seconds=sec),
                    user,
                    USER_IPS[user],
                    pick_action(),
                    pick_status(),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# Attack scenarios
# --------------------------------------------------------------------------- #
def attack_after_hours_login() -> list:
    """Attack 1: 3 successful external logins by EMP_105 outside 9-5."""
    times = [
        ATTACK_DAY + timedelta(hours=2, minutes=15),                # 02:15:00
        ATTACK_DAY + timedelta(hours=2, minutes=15, seconds=45),    # 02:15:45
        ATTACK_DAY + timedelta(hours=23, minutes=40),               # 23:40:00
    ]
    return [make_row(t, "EMP_105", EXTERNAL_IP, "LOGIN", "SUCCESS") for t in times]


def attack_brute_force() -> list:
    """
    Attack 2: 8 failed logins for EMP_108 inside ~2 minutes, then a success.

    Starts at 14:02:10 so the whole burst sits inside one 5-minute window
    (14:00-14:05) and is not split by a window boundary.
    """
    start = ATTACK_DAY + timedelta(hours=14, minutes=2, seconds=10)
    ip = USER_IPS["EMP_108"]
    rows = [
        make_row(start + timedelta(seconds=15 * i), "EMP_108", ip, "LOGIN", "FAILED")
        for i in range(8)  # last failure at 14:03:55
    ]
    rows.append(
        make_row(start + timedelta(seconds=110), "EMP_108", ip, "LOGIN", "SUCCESS")  # 14:04:00
    )
    return rows


def attack_rpm_spike() -> list:
    """Attack 3: 65 API_CALL / DOWNLOAD events by EMP_112 within 11:30:00-11:30:59."""
    ip = USER_IPS[SPIKE_USER]
    seconds = sorted(random.randint(0, 59) for _ in range(65))
    return [
        make_row(
            SPIKE_MINUTE + timedelta(seconds=sec),
            SPIKE_USER,
            ip,
            random.choice(["API_CALL", "DOWNLOAD"]),
            "SUCCESS",
        )
        for sec in seconds
    ]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_dataset(normal_rows: int) -> pd.DataFrame:
    """Combine normal traffic and attacks into one chronologically sorted frame."""
    rows = (
        generate_normal_traffic(normal_rows)
        + attack_after_hours_login()
        + attack_brute_force()
        + attack_rpm_spike()
    )
    df = pd.DataFrame(rows, columns=COLUMNS)
    # ISO-style strings sort chronologically; use a stable sort to keep order on ties
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic system logs.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="normal-traffic rows (default 1000)")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="number of days (default 3)")
    parser.add_argument("--users", type=int, default=DEFAULT_USERS,
                        help=f"number of employees, {MIN_USERS}-{MAX_USERS} (default 20)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="output CSV path")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed (default 42)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not MIN_USERS <= args.users <= MAX_USERS:
        print(f"ERROR: --users must be between {MIN_USERS} and {MAX_USERS}.", file=sys.stderr)
        return 1
    if args.days < 1 or args.rows < 1:
        print("ERROR: --days and --rows must be at least 1.", file=sys.stderr)
        return 1
    # Each (user, minute) slot holds at most 4 events; stay well below capacity
    max_rows = args.days * args.users * MINUTES_PER_DAY * 2
    if args.rows > max_rows:
        print(f"ERROR: {args.rows:,} rows is too many for {args.days} day(s) and {args.users} "
              f"user(s). Max is {max_rows:,}; raise --days or --users.", file=sys.stderr)
        return 1

    random.seed(args.seed)
    if _fake is not None:
        Faker.seed(args.seed)
    configure(args.users, args.days)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)  # create data/ automatically
        df = build_dataset(args.rows)
        df.to_csv(args.output, index=False)
    except OSError as exc:
        print(f"ERROR: could not write {args.output}: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(df):,} log rows to {args.output}")
    print(f"  Normal rows : {args.rows:,}  ({args.users} users, {args.days} day(s))")
    print("  Attacks     : after-hours (3) + brute-force (9) + RPM spike (65)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
