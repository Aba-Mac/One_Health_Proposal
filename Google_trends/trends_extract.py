"""
Google Trends Data Extraction — disease surveillance keywords
=============================================================
Reads a keyword CSV with columns: Keyword, System, Sub-type, Type, Language
Fetches anchor-normalised interest-over-time via trendspy for 6 countries.

Output
------
  trends_keywords.csv    — monthly values per keyword / country
  trends_anchors.csv     — raw anchor keyword values per country

Usage
-----
  pip install trendspy pandas
  python trends_extract.py
"""

import time
import random
import logging
from pathlib import Path

import pandas as pd
from trendspy import Trends


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH  = r"C:\Users\annab\Documents\Work\2026_Senckenberg_Freelance\One_Health_Proposal_Eugenia\Datasets\Keywords.csv"
ENCODING  = "cp1252"
TIMEFRAME = "2010-01-01 2025-12-31"

# One high-frequency anchor term per country (used for cross-batch normalisation).
# Choose a word that is consistently popular so that the anchor signal is stable.
ANCHOR_PER_COUNTRY: dict[str, str] = {
    "DE": "Mücke",
    "AT": "Mücke",
    "CH": "Mücke",
    "IT": "Zanzara",
    "ES": "Mosquito",
    "FR": "Moustique",
}

# Country → language(s) used to filter keywords from the CSV.
# "All" and "Latin" are always appended automatically.
COUNTRY_LANGUAGES: dict[str, list[str]] = {
    "DE": ["German"],
    "AT": ["German"],
    "CH": ["German", "French", "Italian"],
    "IT": ["Italian"],
    "ES": ["Spanish"],
    "FR": ["French"],
}

# Seconds to sleep between API calls — increase if you hit 429 rate-limit errors.
MIN_SLEEP = 15
MAX_SLEEP = 20

# Real keywords per API call (trendspy cap = 5; 1 slot reserved for the anchor).
BATCH_SIZE = 4

OUTPUT_DIR = Path(".")


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────

def load_keywords(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding=ENCODING)
    df.columns = df.columns.str.strip()
    for col in ["Keyword", "Language", "System", "Sub-type", "Type"]:
        df[col] = df[col].astype(str).str.strip()
    log.info("Loaded %d keywords from %s", len(df), csv_path)
    log.info("Sub-types : %s", sorted(df["Sub-type"].unique()))
    log.info("Systems   : %s", sorted(df["System"].unique()))
    return df


def build_country_keyword_lists(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    For each country, combine language-specific keywords with
    All + Latin keywords (deduplicated, order preserved).
    """
    universal = df.loc[
        df["Language"].isin(["All", "Latin"]), "Keyword"
    ].tolist()

    country_kws: dict[str, list[str]] = {}
    for geo, languages in COUNTRY_LANGUAGES.items():
        lang_kws = df.loc[df["Language"].isin(languages), "Keyword"].tolist()
        combined = list(dict.fromkeys(lang_kws + universal))
        country_kws[geo] = combined
        log.info("  %s: %d keywords (%s + All/Latin)", geo, len(combined), languages)
    return country_kws


# ─────────────────────────────────────────────────────────────────────────────
# FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def _sleep(lo: float = MIN_SLEEP, hi: float = MAX_SLEEP) -> None:
    duration = random.uniform(lo, hi)
    log.info("Sleeping %.0fs …", duration)
    time.sleep(duration)


def fetch_batch(
    tr: Trends,
    keywords: list[str],
    anchor: str,
    geo: str,
    timeframe: str,
    max_retries: int = 5,
) -> pd.DataFrame | None:
    """
    Fetch one batch of ≤ BATCH_SIZE keywords + the anchor term.
    """
    kw_list = list(dict.fromkeys(keywords + [anchor]))  # anchor always last
    wait = MIN_SLEEP

    for attempt in range(1, max_retries + 1):
        try:
            df = tr.interest_over_time(kw_list, timeframe=timeframe, geo=geo)
            if df is not None and not df.empty:
                return df
            log.warning(
                "Empty response — %s / %s (attempt %d)", geo, keywords, attempt
            )
        except Exception as exc:
            log.warning(
                "Error attempt %d — %s / %s: %s", attempt, geo, keywords, exc
            )

        jitter = random.uniform(0, wait * 0.3)
        log.info("Back-off %.0fs …", wait + jitter)
        time.sleep(wait + jitter)
        wait = min(wait * 2, 120)

    log.error("All retries exhausted — %s / %s", geo, keywords)
    return None


def fetch_country(
    tr: Trends,
    geo: str,
    keywords: list[str],
    anchor: str,
    timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all keyword batches for one country.

    Returns
    -------
    (keyword_df, anchor_series)
        keyword_df  — anchor-normalised interest, keywords as columns
        anchor_series          — mean raw anchor values across all batches
                                 (useful for diagnosing normalisation quality)
    """
    frames: list[pd.DataFrame] = []
    anchor_series_list: list[pd.Series] = []

    for i in range(0, len(keywords), BATCH_SIZE):
        batch = keywords[i : i + BATCH_SIZE]
        log.info("  Fetching %s batch %d/%d: %s",
                 geo, i // BATCH_SIZE + 1,
                 -(-len(keywords) // BATCH_SIZE),  # ceiling division
                 batch)
        df = fetch_batch(tr, batch, anchor, geo, timeframe)

        if df is None:
            # No sleep here — fetch_batch already back-offs internally.
            continue

        # Drop the isPartial helper column BEFORE accessing keyword columns
        # so that keyword names are never accidentally removed.
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])

        # Anchor is included in every batch, so Google Trends has already
        # put the keywords and anchor on the same 0–100 scale.
        anchor_series = df[anchor].copy()
        anchor_series.name = f"batch_{i // BATCH_SIZE + 1}"
        anchor_series_list.append(anchor_series)

        batch_num = i // BATCH_SIZE + 1
        keyword_df = df.drop(columns=[anchor], errors="ignore")
        keyword_df = keyword_df.rename(columns=lambda c: f"{c}::batch_{batch_num}")
        frames.append(keyword_df)
        _sleep()  # polite pause between batches

    # Guard: nothing was collected
    if not frames:
        log.warning("No data collected for %s", geo)
        return (pd.DataFrame(), pd.DataFrame())

    result = pd.concat(frames, axis=1)
    result.index = pd.to_datetime(result.index)

    anchor_df = pd.concat(anchor_series_list, axis=1)
    anchor_df.index = pd.to_datetime(anchor_df.index)

    return result.sort_index(), anchor_df.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load CSV
    kw_df = load_keywords(CSV_PATH)

    # 2. Build per-country keyword lists (language-specific + All + Latin)
    country_kws = build_country_keyword_lists(kw_df)

    # 3. Fetch from Google Trends
    tr = Trends(request_delay=6.0)

    country_data: dict[str, pd.DataFrame] = {}
    anchor_data: dict[str, pd.DataFrame] = {}

    for geo, keywords in country_kws.items():
        anchor = ANCHOR_PER_COUNTRY[geo]
        log.info("── Fetching country: %s (anchor: %s) ──", geo, anchor)
        df, anchor_series = fetch_country(tr, geo, keywords, anchor, TIMEFRAME)

        if not df.empty:
            country_data[geo] = df
            anchor_data[geo]  = anchor_series
        else:
            log.warning("Skipping %s — no data returned.", geo)

        # Longer pause between countries to avoid sustained rate-limiting
        _sleep(lo=MAX_SLEEP, hi=MAX_SLEEP * 2)

    if not country_data:
        log.error("No data collected for any country. Check your API access and retry.")
        return

    # 4. Save raw anchor values
    anchor_frames = []

    for geo, df in anchor_data.items():
        tmp = df.copy()
        tmp.columns = [
            f"{geo}::{ANCHOR_PER_COUNTRY[geo]}::{col}"
            for col in tmp.columns
        ]
        anchor_frames.append(tmp)

    anchor_df = pd.concat(anchor_frames, axis=1)

    anchor_path = OUTPUT_DIR / "trends_anchors.csv"
    anchor_df.to_csv(anchor_path)
    log.info("Anchor data → %s", anchor_path)

    # 5. Save raw keyword-level data with "GEO::keyword" columns
    raw_frames = []
    for geo, df in country_data.items():
        tmp = df.copy()
        tmp.columns = [f"{geo}::{col}" for col in tmp.columns]
        raw_frames.append(tmp)

    raw_all  = pd.concat(raw_frames, axis=1)
    raw_path = OUTPUT_DIR / "trends_keywords.csv"
    raw_all.to_csv(raw_path)
    log.info("Keyword data saved → %s", raw_path)

    log.info("Extraction complete ✓  Next step: run trends_analysis.ipynb")


if __name__ == "__main__":
    main()
