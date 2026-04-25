"""Download and convert the selected NYC DCP ACS NTA dataset to CSV."""

from __future__ import annotations

import os
import time
import zipfile
from io import BytesIO

import pandas as pd
import requests

SOURCE_ZIP_URL = (
    "https://s-media.nyc.gov/agencies/dcp/assets/files/zip/"
    "data-tools/population/american-community-survey/5-yr-ACS-2023.zip"
)
ZIP_MEMBER_PATH = (
    "5-yr ACS 2023/5-yr ACS 2023/Neighborhood-NTA/Economic/Econ_1923_NTA.xlsx"
)
RAW_XLSX_PATH = "data/raw/Econ_1923_NTA.xlsx"
CSV_OUTPUT_PATH = "data/raw/census_nta.csv"
LOG_OUTPUT_PATH = "data/raw/census_download_log.txt"
TARGET_SHEET = "EconData"
REQUEST_TIMEOUT = 120


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def download_zip_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def extract_member(zip_bytes: bytes, member_path: str) -> bytes:
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        return archive.read(member_path)


def choose_sheet(sheet_names: list[str]) -> str:
    if TARGET_SHEET in sheet_names:
        return TARGET_SHEET
    for sheet_name in sheet_names:
        if "data" in sheet_name.lower():
            return sheet_name
    return sheet_names[0]


def assess_columns(columns: list[str]) -> tuple[bool, bool, bool]:
    lower_map = {column.lower(): column for column in columns}
    has_identifier = any(key in lower_map for key in ("geoid", "geogname", "ntatype"))
    has_median_income = any("mdhhinc" in column.lower() for column in columns)
    has_population = any(
        token in column.lower()
        for token in ("pop_1e", "population", "pop16ple")
        for column in columns
    )
    return has_identifier, has_median_income, has_population


def build_log(
    *,
    source_url: str,
    downloaded_type: str,
    chosen_sheet: str,
    chosen_reason: str,
    columns: list[str],
    row_count: int,
    has_identifier: bool,
    has_median_income: bool,
    has_population: bool,
) -> str:
    return "\n".join(
        [
            f"download_timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"source_url: {source_url}",
            f"source_member_within_zip: {ZIP_MEMBER_PATH}",
            f"file_type_downloaded: {downloaded_type}",
            f"sheet_name_used: {chosen_sheet}",
            f"why_this_file_was_chosen: {chosen_reason}",
            f"row_count: {row_count}",
            f"column_count: {len(columns)}",
            f"nta_or_neighborhood_identifier_present: {has_identifier}",
            (
                "median_income_present: "
                f"{has_median_income} (matched on columns like MdHHIncE)"
            ),
            (
                "population_present: "
                f"{has_population} (this workbook includes Pop16plE, a 16+ population field; "
                "it does not include the Demographic workbook's total population field Pop_1E)"
            ),
            "column_names:",
            *columns,
            "",
            "notes:",
            (
                "The NYC DCP ACS 2023 5-year NTA release is split into separate thematic workbooks. "
                "The Economic workbook was selected because it is recent, NTA-level, and contains "
                "median household income along with neighborhood identifiers and other economic fields."
            ),
            (
                "An alternative candidate, Dem_1923_NTA.xlsx, contains total population but not median "
                "household income, so it was not chosen as the primary single-file download for this phase."
            ),
        ]
    )


def main() -> None:
    ensure_parent(RAW_XLSX_PATH)
    ensure_parent(CSV_OUTPUT_PATH)
    ensure_parent(LOG_OUTPUT_PATH)

    zip_bytes = download_zip_bytes(SOURCE_ZIP_URL)
    workbook_bytes = extract_member(zip_bytes, ZIP_MEMBER_PATH)

    with open(RAW_XLSX_PATH, "wb") as workbook_file:
        workbook_file.write(workbook_bytes)

    excel_file = pd.ExcelFile(RAW_XLSX_PATH)
    chosen_sheet = choose_sheet(excel_file.sheet_names)
    frame = pd.read_excel(RAW_XLSX_PATH, sheet_name=chosen_sheet)
    frame.to_csv(CSV_OUTPUT_PATH, index=False)

    columns = [str(column) for column in frame.columns.tolist()]
    has_identifier, has_median_income, has_population = assess_columns(columns)
    looks_usable = has_identifier and has_median_income
    if has_population and not any(column.lower() == "pop_1e" for column in columns):
        looks_usable = looks_usable and True

    chosen_reason = (
        "Selected the 2019-2023 ACS 5-year Neighborhood-NTA Economic workbook because it is the "
        "most recent NTA-level file on the official NYC DCP ACS page that includes median household income "
        "and neighborhood identifiers in a single workbook."
    )
    log_text = build_log(
        source_url=SOURCE_ZIP_URL,
        downloaded_type="xlsx (extracted from official zip release)",
        chosen_sheet=chosen_sheet,
        chosen_reason=chosen_reason,
        columns=columns,
        row_count=len(frame),
        has_identifier=has_identifier,
        has_median_income=has_median_income,
        has_population=has_population,
    )
    with open(LOG_OUTPUT_PATH, "w", encoding="utf-8") as log_file:
        log_file.write(log_text + "\n")

    print("files created:")
    print(RAW_XLSX_PATH)
    print(CSV_OUTPUT_PATH)
    print(LOG_OUTPUT_PATH)
    print("column names:")
    print(columns)
    print("first 10 rows:")
    print(frame.head(10).to_string(index=False))
    print("looks usable for later merge steps:")
    if has_identifier and has_median_income and has_population:
        print(
            "yes, with the caveat that the population field here is Pop16plE rather than total population."
        )
    else:
        print("no, this file is missing one or more core fields needed later.")


if __name__ == "__main__":
    main()
