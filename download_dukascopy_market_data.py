"""
Download spot XAG/USD (silver), XAU/USD (gold) and EUR/USD 1-minute data from
Dukascopy, merge the chunks, and build 5m / 15m / 1h candles.  (v2, Windows-safe)

Requirements (once):
    - Node.js 18 or higher  (https://nodejs.org)
    - pip install pandas

Run from the project root:
    py download_dukascopy_market_data.py
    py download_dukascopy_market_data.py --start 2023-11-01 --end 2026-11-14

RESUMABLE: finished chunks are kept in data/raw/dukascopy/chunks/. If anything
fails, just run it again and it continues where it stopped.

Output (UTC index, columns open/high/low/close; spot volume is intentionally excluded):
    data/raw/dukascopy/<instrument>_1m.csv, _5m.csv, _15m.csv, _1h.csv
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_INSTRUMENTS = ["xagusd", "xauusd", "eurusd"]
OUT = Path("data/raw/dukascopy")
CHUNKS = OUT / "chunks"
RESAMPLE = {"5m": "5min", "15m": "15min", "1h": "1h"}
CHUNK_DAYS = 60      # smaller chunks are more reliable on Windows
RETRIES = 5
PAUSE_BETWEEN_CHUNKS = 2


def safe_rmtree(path: Path):
    """Delete a folder; never raise (Windows may keep a file handle briefly)."""
    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except Exception:
            time.sleep(1)
    shutil.rmtree(path, ignore_errors=True)


def chunk_ranges(start: date, end: date, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def run_chunk(npx: str, inst: str, a: date, b: date):
    """Download one chunk; return the CSV path or None. Never raises."""
    target = CHUNKS / inst / f"{inst}_{a}_{b}.csv"
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    cmd = [npx, "--yes", "dukascopy-node", "-i", inst,
           "-from", str(a), "-to", str(b), "-t", "m1", "-f", "csv"]

    for attempt in range(1, RETRIES + 1):
        # Fresh SHORT temp folder (system temp) for every attempt.
        work = Path(tempfile.mkdtemp(prefix="dk_"))
        try:
            res = subprocess.run(cmd, cwd=work, capture_output=True,
                                 text=True, timeout=60 * 30)
            files = [f for f in work.rglob("*.csv") if f.stat().st_size > 0]
            if res.returncode == 0 and files:
                shutil.copyfile(files[0], target)
                return target
            lines = (res.stderr or res.stdout).strip().splitlines()
            print(f"    attempt {attempt} failed: {(lines[-1] if lines else '')[:110]}")
        except subprocess.TimeoutExpired:
            print(f"    attempt {attempt} timed out")
        except Exception as e:
            print(f"    attempt {attempt} error: {str(e)[:110]}")
        finally:
            safe_rmtree(work)
        time.sleep(5 * attempt)
    return None


def merge_instrument(inst: str) -> pd.DataFrame:
    parts = []
    for f in sorted((CHUNKS / inst).glob(f"{inst}_*.csv")):
        df = pd.read_csv(f)
        if df.empty:
            continue
        df.columns = [c.strip().lower() for c in df.columns]
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    return df[cols].astype(float)


def resample(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Candle stamped T covers [T, T+rule) and is only known at T+rule."""
    n_needed = int(pd.Timedelta(rule) / pd.Timedelta("1min"))
    rules = {"open": "first", "high": "max", "low": "min", "close": "last"}
    rules = {c: f for c, f in rules.items() if c in df_1m.columns}  # volume may be absent
    agg = df_1m.resample(rule, closed="left", label="left").agg(rules)
    count = df_1m["close"].resample(rule, closed="left", label="left").count()
    return agg[count == n_needed]        # complete candles only


def ensure_output_paths(out_dir: Path, instruments: list[str], *, force: bool = False) -> None:
    """Refuse final-file replacement unless the operator explicitly opts in."""
    existing = [out_dir / f"{inst}_{name}.csv" for inst in instruments for name in ("1m", "5m", "15m", "1h")]
    existing = [path for path in existing if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing[:5])
        suffix = "..." if len(existing) > 5 else ""
        raise FileExistsError(f"Refusing to overwrite existing CSVs: {names}{suffix}; pass --force to replace them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-11-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--instruments", nargs="+", default=DEFAULT_INSTRUMENTS)
    ap.add_argument("--out-dir", default="data/raw/dukascopy", help="Output directory for chunks and CSVs.")
    ap.add_argument("--force", action="store_true", help="Allow replacing existing final CSVs.")
    args = ap.parse_args()

    global OUT, CHUNKS
    OUT = Path(args.out_dir)
    CHUNKS = OUT / "chunks"
    try:
        ensure_output_paths(OUT, args.instruments, force=args.force)
    except FileExistsError as exc:
        sys.exit(str(exc))

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    npx = shutil.which("npx")
    if npx is None:
        sys.exit("npx not found. Install Node.js 18+ from https://nodejs.org and retry.")

    OUT.mkdir(parents=True, exist_ok=True)
    failed = []

    for inst in args.instruments:
        print(f"\n=== {inst} ===")
        for a, b in chunk_ranges(start, end, CHUNK_DAYS):
            print(f"  {a} -> {b}")
            if run_chunk(npx, inst, a, b) is None:
                failed.append((inst, a, b))
                print("    FAILED (run the script again to retry)")
            time.sleep(PAUSE_BETWEEN_CHUNKS)

        df = merge_instrument(inst)
        if df.empty:
            print(f"  no data merged for {inst}")
            continue
        df.to_csv(OUT / f"{inst}_1m.csv")
        print(f"  1m : {len(df):>9,} rows  {df.index.min()} -> {df.index.max()}")
        for name, rule in RESAMPLE.items():
            r = resample(df, rule)
            r.to_csv(OUT / f"{inst}_{name}.csv")
            print(f"  {name:3s}: {len(r):>9,} rows")

    print("\n=== SUMMARY ===")
    if failed:
        print(f"{len(failed)} chunk(s) failed. Run the script again to retry:")
        for inst, a, b in failed:
            print(f"  {inst} {a} -> {b}")
    else:
        print("All chunks downloaded. Row counts should be similar across instruments.")


if __name__ == "__main__":
    main()
