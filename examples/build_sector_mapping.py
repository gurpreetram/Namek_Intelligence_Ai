"""
Builds config/sector_mapping.json from OFFICIAL NSE sectoral index data
(niftyindices.com) -- not guessed/memorized sector assignments, which is
exactly the kind of thing that produces silent misclassification errors.

Two data sources, in priority order:
  1. Full constituent CSV (niftyindices.com/IndexConstituent/*.csv) -- the
     REAL, complete official membership list (10-30 stocks per index,
     depending on the sector), with a `Symbol` column that already IS the
     NSE trading symbol -- an exact match against Upstox's instrument
     master, no fuzzy company-name matching needed. Used for 14 of 15
     sectors.
  2. Factsheet PDF's "Top constituents by weightage" table -- only the
     TOP 10 BY WEIGHT, not the full membership, and requires fuzzy
     company-name matching (see match_to_trading_symbol()). Used ONLY for
     FINANCIAL_SERVICES, whose full-constituent CSV URL could not be
     located despite extensive guessing (many slug variations tried and
     failed) -- a real, acknowledged gap in this run, not silently
     patched over.

Honest limitation, not a bug: sectoral indices themselves only cover
NSE's largest/most liquid ~15-30 stocks per sector (drawn from Nifty 500).
Most small/mid-caps genuinely have no official sector classification
available this way -- a real, permanent constraint on coverage, not
something to fix by guessing.

Overlap handling: a stock can legitimately appear in multiple sectoral
indices (e.g. HDFC Bank in Nifty Bank AND Nifty Private Bank AND Nifty
Financial Services). This script processes sectors in SECTOR_SOURCES's
iteration order and the LAST sector processed wins on overlap -- a known
limitation carried over from the original version of this mapping, not a
deliberate "primary sector" choice.

Run: python -m examples.build_sector_mapping
"""
from __future__ import annotations

import difflib
import io
import json
import re
from dataclasses import dataclass

import pdfplumber
import requests

from config.settings import PROJECT_ROOT
from data.instruments import load_instrument_master

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


@dataclass
class SectorSource:
    kind: str  # "csv" | "pdf"
    url: str


# All 15 TRUE sectoral indices (companies grouped by industry) requested.
# Deliberately excludes thematic/cross-cutting indices found along the way
# (Nifty Commodities, Infrastructure, MNC, PSE, CPSE, Services Sector,
# India Consumption) -- those aren't sectors, they're investment themes
# that span multiple sectors, and merging them in would corrupt the
# "primary sector" concept even further than the existing overlap issue
# already does (see module docstring).
SECTOR_SOURCES: dict[str, SectorSource] = {
    "IT": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftyitlist.csv"),
    "AUTO": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftyautolist.csv"),
    "PHARMA": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftypharmalist.csv"),
    "FMCG": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftyfmcglist.csv"),
    "METAL": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftymetallist.csv"),
    "REALTY": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftyrealtylist.csv"),
    "MEDIA": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftymedialist.csv"),
    "ENERGY": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftyenergylist.csv"),
    "OIL_GAS": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftyoilgaslist.csv"),
    "CONSUMER_DURABLES": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftyconsumerdurableslist.csv"),
    "HEALTHCARE": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftyhealthcarelist.csv"),
    "PSU_BANK": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftypsubanklist.csv"),
    "PRIVATE_BANK": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_nifty_privatebanklist.csv"),
    "BANK": SectorSource("csv", "https://www.niftyindices.com/IndexConstituent/ind_niftybanklist.csv"),
    # PDF fallback -- full-constituent CSV URL not found for this one.
    "FINANCIAL_SERVICES": SectorSource("pdf", "https://www.niftyindices.com/Factsheet/ind_Nifty_Financial_Services.pdf"),
}

# Manual overrides for names heavily abbreviated in Upstox's instrument
# master vs. the factsheet's full legal name -- only needed for the PDF
# (fuzzy name matching) path. The CSV path matches on Symbol directly, no
# name fuzziness involved.
MANUAL_OVERRIDES = {
    "Bajaj Finance Ltd.": "BAJFINANCE",
    "Bajaj Finserv Ltd.": "BAJAJFINSV",
    "Shriram Finance Ltd.": "SHRIRAMFIN",
    "SBI Life Insurance Company Ltd.": "SBILIFE",
    "BSE Ltd.": "BSE",
}

CONSTITUENTS_HEADER_RE = re.compile(r"Top constituents by weightage", re.IGNORECASE)
CONSTITUENT_LINE_RE = re.compile(r"^(.+?)\s+([\d]+\.\d+)$")


def fetch(url: str) -> bytes:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.content


def parse_csv_symbols(csv_bytes: bytes) -> list[str]:
    text = csv_bytes.decode("utf-8-sig", errors="replace")
    lines = [l for l in text.splitlines() if l.strip()]
    header = [h.strip() for h in lines[0].split(",")]
    symbol_idx = header.index("Symbol")
    return [l.split(",")[symbol_idx].strip() for l in lines[1:]]


def parse_pdf_constituents(pdf_bytes: bytes) -> list[tuple[str, float]]:
    """
    Regex-based extraction directly on page.extract_text(), NOT
    pdfplumber's extract_tables() -- confirmed bug: extract_tables() drops
    the #1-weighted constituent (the highest-weight row) on every sector
    due to a layout quirk merging/misplacing the first row.
    """
    constituents: list[tuple[str, float]] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text or not CONSTITUENTS_HEADER_RE.search(text):
                continue
            lines = text.split("\n")
            start = next(i for i, l in enumerate(lines) if CONSTITUENTS_HEADER_RE.search(l))
            for line in lines[start + 2:]:
                m = CONSTITUENT_LINE_RE.match(line.strip())
                if not m:
                    break
                constituents.append((m.group(1).strip(), float(m.group(2))))
            break
    return constituents


def match_company_name(company_name: str, master_df, cutoff: float = 0.82) -> str | None:
    if company_name in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[company_name]
    names = master_df["name"].dropna().tolist()
    matches = difflib.get_close_matches(company_name.upper(), [n.upper() for n in names], n=1, cutoff=cutoff)
    if not matches:
        return None
    row = master_df[master_df["name"].str.upper() == matches[0]]
    return str(row.iloc[0]["trading_symbol"]) if not row.empty else None


def main() -> None:
    master_df = load_instrument_master()
    valid_symbols = set(master_df["trading_symbol"])

    mapping: dict[str, str] = {}
    unmatched: list[tuple[str, str]] = []
    per_sector_counts: dict[str, int] = {}

    for sector, source in SECTOR_SOURCES.items():
        print(f"Fetching {sector} ({source.kind}: {source.url})...")
        try:
            raw = fetch(source.url)
        except Exception as e:
            print(f"  FAILED to fetch: {e}")
            continue

        matched_count = 0
        if source.kind == "csv":
            symbols = parse_csv_symbols(raw)
            print(f"  {len(symbols)} constituents in official list")
            for symbol in symbols:
                if symbol in valid_symbols:
                    mapping[symbol] = sector
                    matched_count += 1
                else:
                    unmatched.append((sector, symbol))
        else:  # pdf
            constituents = parse_pdf_constituents(raw)
            print(f"  {len(constituents)} constituents in factsheet's top-10-by-weight table")
            for company_name, _weight in constituents:
                symbol = match_company_name(company_name, master_df)
                if symbol is None:
                    unmatched.append((sector, company_name))
                    continue
                mapping[symbol] = sector
                matched_count += 1

        per_sector_counts[sector] = matched_count

    output_path = PROJECT_ROOT / "config" / "sector_mapping.json"
    output_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))

    print()
    print("=" * 70)
    print("SECTOR MAPPING BUILD RESULT")
    print("=" * 70)
    print(f"Sectors covered: {len(per_sector_counts)}/{len(SECTOR_SOURCES)}")
    for sector, count in per_sector_counts.items():
        print(f"  {sector:20s} {count} symbols matched")
    print(f"\nTotal unique symbols mapped: {len(mapping)}")
    print(f"Unmatched ({len(unmatched)}):")
    for sector, name in unmatched:
        print(f"  {sector}: {name!r}")
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
