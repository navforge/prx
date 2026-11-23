import logging
import math
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import re

from typing_extensions import NamedTuple

from prx.util import (
    hash_of_file_content,
    timeit,
    disk_cache,
    try_repair_with_gfzrnx,
)

logger = logging.getLogger(__name__)

CONSTELLATIONS = ["G", "E", "C", "R", "I", "J", "S"]

RECORD_FIELDS = [
    ["A1,I2.2", "1XI4", 5*"1X,I2.2", 3*"D19.12"]  # SV / EPOCH / SV CLK row
]

def convert_nav_dataset_to_dataframe(nav_ds):
    """convert ephemerides from xarray.Dataset to pandas.DataFrame"""
    df = nav_ds.to_dataframe()
    # Drop ephemerides for which all parameters are NaN, as we cannot compute anything from those
    df = df.dropna(how="all")
    df = df.reset_index()
    # georinex adds suffixes to satellite IDs if it sees multiple ephemerides (e.g. F/NAV, I/NAV) for the same
    # satellite and the same timestamp.
    # The downstream code expects three-letter satellite IDs, so remove suffixes.
    df["sv"] = df.sv.str[:3]
    df["constellation"] = df["sv"].str[0]
    df["time_scale"] = df["constellation"].replace(
        constants.constellation_2_system_time_scale
    )

    # Keep only SBAS ephemerides with low URA, these are based on SBAS message type 9, while those with large URA are
    # based on message type 17 which only contains almanac-like coarse satellite position estimates.
    if "URA" in df.columns:
        df = df[~((df.sv.str.startswith("S")) & (df.URA >= constants.cSbasURALimit))]

    def compute_ephemeris_and_clock_offset_reference_times(group):
        week_field = {
            "GPST": "GPSWeek",
            "GST": "GALWeek",
            "BDT": "BDTWeek",
            "SBAST": "GPSWeek",
            "QZSST": "GPSWeek",
            "IRNSST": "GPSWeek",
        }
        group_time_scale = group["time_scale"].iloc[0]
        if group_time_scale not in ["GLONASST", "SBAST"]:
            full_seconds = (
                group[week_field[group_time_scale]] * constants.cSecondsPerWeek
                + group["Toe"]
            )
            group["ephemeris_reference_time_system_time"] = (
                constants.system_time_scale_rinex_utc_epoch[group_time_scale]
                + pd.to_timedelta(full_seconds, unit="seconds")
            )
        else:
            # For SBAS and GLONASS there are no separate ephemeris reference time fields
            group["ephemeris_reference_time_system_time"] = group["time"]
            # The first derivative of the clock offset is in a different field for SBAS and GLONASS
            group["SVclockDrift"] = group["SVrelFreqBias"]
            # And the second derivative is zero, i.e. the constellation ground segment uses a fist-order clock model
            group["SVclockDriftRate"] = 0
        group["ephemeris_reference_time_isagpst"] = to_isagpst(
            group["ephemeris_reference_time_system_time"],
            group_time_scale,
            int(nav_ds.attrs["utc_gpst_leap_seconds"]),
        )
        group["clock_offset_reference_time_system_time"] = group["time"]
        group["clock_reference_time_isagpst"] = to_isagpst(
            group["clock_offset_reference_time_system_time"],
            group_time_scale,
            int(nav_ds.attrs["utc_gpst_leap_seconds"]),
        )
        return group

    df = (
        df.groupby("constellation")
        .apply(compute_ephemeris_and_clock_offset_reference_times)
        .reset_index(drop=True)
    )
    df = set_time_of_validity(df)
    df = df.rename(
        columns={
            "M0": "M_0",
            "Eccentricity": "e",
            "Toe": "t_oe",
            "DeltaN": "deltaN",
            "Cuc": "C_uc",
            "Cus": "C_us",
            "Cic": "C_ic",
            "Crc": "C_rc",
            "Cis": "C_is",
            "Crs": "C_rs",
            "Io": "i_0",
            "Omega0": "Omega_0",
        },
    )
    df["orbit_type"] = df.constellation.map(
        {
            "C": "kepler",
            "G": "kepler",
            "E": "kepler",
            "J": "kepler",
            "S": "sbas",
            "R": "glonass",
            "I": "irnss",
        }
    )
    df = df.reset_index(drop=True)
    df = compute_gal_inav_fnav_indicators(df)
    if "R" in df.constellation.unique():
        df["frequency_slot"] = df.FreqNum.where(df.sv.str[0] == "R", 1).astype(int)
    df.attrs["ionospheric_corr_GPS"] = nav_ds.ionospheric_corr_GPS
    return df


def lazy_read(rinex_nav_file_path: Path) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    df = pl.scan_csv(
        rinex_nav_file_path,
        has_header=False,
        separator="\x1f",
        quote_char=None,
        new_columns=["lines"],
        infer_schema=False,
        missing_utf8_is_empty_string=True
    ).with_row_index("i")
    last_header_row = (
        df.filter(pl.col("lines").str.contains("END OF HEADER"))
        .select(pl.col("i").first()).collect()
        .item()
    )
    return df.slice(0, last_header_row), df.slice(last_header_row)

def parse_header(rinex_nav_file_path: Path) -> pl.DataFrame:
    lazy_header, _ = lazy_read(rinex_nav_file_path)
    df = lazy_header.collect()
    return df

def parse_records(rinex_nav_file_path: Path) -> pl.DataFrame:
    _, lazy_records = lazy_read(rinex_nav_file_path)
    df = lazy_records.collect().drop("i")
    # Assign record ID and row number within the record
    df = df.with_columns(record_id=pl.col("lines").str.slice(0, 1).is_in(CONSTELLATIONS).cum_sum()).filter(pl.col("record_id") > 0).with_columns(
            block_row=pl.int_range(0, pl.len()).over("record_id")
        )
    df = df.pivot(
            on="block_row",
            index="record_id",
            values="lines",
    ).drop("record_id")
    # Split into fields, these are the same for all constellations, although
    # they can mean different things
    df = df.with_columns(
        ""
        pl.col("0").str.split(0,1).alias("system"),
        pl.col("0").str.split(1,2).alias("prn"),
        pl.col("0").str.split(3,4).alias("year"),
        pl.col("0").str.split(9,2).alias("month"),
        pl.col("0").str.split(9,2).alias("month"),

    )
    return df



def parse(file_path):
    header = parse_header(file_path)
    records = parse_records(file_path)
    return 0
