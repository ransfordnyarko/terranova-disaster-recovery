import pandas as pd
import numpy as np
from src.utils.helpers import get_logger
from pathlib import Path

log = get_logger(__name__)

BASE_URL = "https://www.fema.gov/api/open"

DOWNLOAD_URLS = {
    "declarations":       f"{BASE_URL}/v2/DisasterDeclarationsSummaries.csv",
    "public_assistance":  f"{BASE_URL}/v2/PublicAssistanceFundedProjectsDetails.csv",
    "disaster_summaries": f"{BASE_URL}/v1/FemaWebDisasterSummaries.csv",
}

FIELDS = {
    "declarations": [
        "disasterNumber",
        "state",
        "declarationDate",
        "incidentType",
        "incidentBeginDate",
        "incidentEndDate",
        "declarationType",
        "ihProgramDeclared",
        "iaProgramDeclared",
        "paProgramDeclared",
        "hmProgramDeclared",
        "designatedArea",
        "fyDeclared",
    ],
    "public_assistance": [
        "disasterNumber",
        "stateAbbreviation",
        "incidentType",
        "obligatedAmount",
        "projectCategory",
        "projectAmount",
        "federalShareObligated",
        "totalObligated",
        "projectSize",
        "damageCategoryCode",
        "applicantId",
    ],
    "disaster_summaries": [
        "disasterNumber",
        "totalAmountIhpApproved",
        "totalAmountHaApproved",
        "totalAmountOnaApproved",
        "totalObligatedAmountPa",
        "totalObligatedAmountCatAb",
        "totalObligatedAmountCatC2g",
        "totalObligatedAmountHmgp",
        "paLoadDate",
        "iaLoadDate",
    ],
}


def _fetch_bulk_csv(name: str) -> pd.DataFrame:
    url = DOWNLOAD_URLS[name]
    fields = FIELDS[name]

    log.info(f"Downloading {name} from {url} ...")
    df = pd.read_csv(url, low_memory=False)
    log.info(f"Downloaded {len(df):,} rows for '{name}'")

    # Keep only columns that actually exist in the download
    available = [f for f in fields if f in df.columns]
    missing = set(fields) - set(available)
    if missing:
        log.warning(f"'{name}': columns not found in download and will be skipped: {missing}")

    return df[available]


def fetch_declarations() -> pd.DataFrame:
    return _fetch_bulk_csv("declarations")

def fetch_public_assistance() -> pd.DataFrame:
    return _fetch_bulk_csv("public_assistance")

def fetch_disaster_summaries() -> pd.DataFrame:
    return _fetch_bulk_csv("disaster_summaries")


RAW_DATA_DIR = Path("__file__").resolve().parent/ "data" / "raw"

def save_data(df: pd.DataFrame, filename: str):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_DATA_DIR / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    log.info(f"Saved {len(df):,} rows to {filepath}")


def run_ingestion():
    log.info("Starting data ingestion...")

    decl = fetch_declarations()
    PA   = fetch_public_assistance()
    DS   = fetch_disaster_summaries()

    save_data(decl, "declarations")
    save_data(PA,   "public_assistance")
    save_data(DS,   "disaster_summaries")

    log.info("Data ingestion completed.")


if __name__ == "__main__":
    run_ingestion()