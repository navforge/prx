import logging
from functools import lru_cache
from pathlib import Path

import polars as pl
import re

from prx.util import (
    hash_of_file_content,
    timeit,
    disk_cache,
    try_repair_with_gfzrnx,
)

logger = logging.getLogger(__name__)

# See table A3 in the RINEX 3.05 specification
obs_length = 14
block_length = obs_length + 1 + 1
sat_prefix_length = 3


def read_lines(file_path):
    # Use a separator that does not occur in RINEX files, so that each line ends up in a single column
    return pl.read_csv(
        file_path,
        has_header=False,
        separator="\x1f",
        quote_char=None,
        new_columns=["lines"],
        infer_schema=False,
    ).with_row_index("i")


def get_obs_types(header):
    types = []
    marker = "SYS / # / OBS TYPES"
    lines = " " + " ".join(
        header.filter(pl.col("lines").str.contains(marker, literal=True))[
            "lines"
        ].to_list()
    ).replace(marker, " ")
    blocks = re.split(r"( C | E | J | R | I | G | S )", lines)
    blocks = [block.strip() for block in blocks if len(block.strip()) > 0]
    blocks = [
        block + " " + blocks[i + 1] for i, block in enumerate(blocks) if len(block) == 1
    ]
    for block in blocks:
        types.append(
            {
                "constellation": block.split()[0].strip(),
                "number_of_sats": int(block.split()[1].strip()),
                "obs_types": [element.strip() for element in block.split()[2:]],
            }
        )
    return {content["constellation"]: content for content in types}


def get_glonass_slot(file_path):
    df = read_lines(file_path)
    i_end_of_header = df.filter(
        pl.col("lines").str.contains("END OF HEADER", literal=True)
    )["i"][0]
    header = df.filter(pl.col("i") < i_end_of_header)
    marker = "GLONASS SLOT / FRQ #"
    lines = " " + " ".join(
        header.filter(pl.col("lines").str.contains(marker, literal=True))[
            "lines"
        ].to_list()
    ).replace(marker, " ")
    return {int(prn_slot[:2]): int(prn_slot[2:]) for prn_slot in lines.split("R")[1:]}


def get_obs_type_mapping(obs_types):
    # Maps (constellation, column of the record) to the observation type stored in that column
    mapping = []
    for constellation, content in obs_types.items():
        for i, obs_type in enumerate(content["obs_types"]):
            mapping.append(
                {
                    "constellation": constellation,
                    "column": f"obs_{i}",
                    "obs_type": obs_type,
                }
            )
            if obs_type[0] == "L":
                mapping.append(
                    {
                        "constellation": constellation,
                        "column": f"lli_{i}",
                        "obs_type": obs_type + "lli",
                    }
                )
    return pl.DataFrame(
        mapping,
        schema={"constellation": pl.String, "column": pl.String, "obs_type": pl.String},
    )


def parse(file_path):
    df = read_lines(file_path)
    i_end_of_header = df.filter(
        pl.col("lines").str.contains("END OF HEADER", literal=True)
    )["i"][0]
    header = df.filter(pl.col("i") < i_end_of_header)
    obs_types = get_obs_types(header)
    n_columns = max(len(content["obs_types"]) for content in obs_types.values())

    def obs_start(i):
        return sat_prefix_length + i * block_length

    is_timestamp = pl.col("lines").str.starts_with(">")
    records = (
        df.lazy()
        .filter(pl.col("i") > i_end_of_header)
        .with_columns(
            time=pl.when(is_timestamp)
            .then(
                pl.col("lines")
                .str.slice(2, 27)
                .str.strptime(pl.Datetime("ns"), "%Y %m %d %H %M %S%.f", strict=False)
            )
            .forward_fill(),
            # See table A2 in the RINEX 3.05 specification: epoch flags 0 (OK) and 1 (power failure) are followed
            # by observation records. Epoch flags 2 to 5 are followed by special records (header lines), epoch flag 6
            # by cycle slip records, which we both skip.
            epoch_flag=pl.when(is_timestamp)
            .then(pl.col("lines").str.slice(31, 1))
            .forward_fill()
            .cast(pl.Int64),
        )
        .filter(~is_timestamp & (pl.col("epoch_flag") <= 1))
        .select(
            "time",
            sv=pl.col("lines").str.slice(0, sat_prefix_length),
            *[
                pl.col("lines")
                .str.slice(obs_start(i), obs_length)
                .str.strip_chars()
                .replace("", None)
                .cast(pl.Float64)
                .alias(f"obs_{i}")
                for i in range(n_columns)
            ],
            # retrieve loss of lock indicator
            *[
                pl.col("lines")
                .str.slice(obs_start(i) + obs_length, 1)
                .str.strip_chars()
                .replace("", "0")
                .cast(pl.Int64)
                .cast(pl.Float64)
                .alias(f"lli_{i}")
                for i in range(n_columns)
            ],
        )
    )
    result = (
        records.unpivot(
            index=["time", "sv"], variable_name="column", value_name="obs_value"
        )
        .with_columns(constellation=pl.col("sv").str.slice(0, 1))
        .join(get_obs_type_mapping(obs_types).lazy(), on=["constellation", "column"])
        .filter(pl.col("obs_value").is_not_null())
        .select("time", "sv", "obs_value", "obs_type")
        .sort("time", maintain_order=True)
        .collect()
    )
    return result


@lru_cache
@disk_cache.cache(ignore=["rinex_file"])
def parse_rinex_obs_file_cached(rinex_file: Path, file_hash: str):
    logger.info(f"Parsing {rinex_file} (hash {file_hash}) ...")
    try_repair_with_gfzrnx(rinex_file)
    return parse(rinex_file)


@timeit
def parse_rinex_obs_file(rinex_file_path: Path):
    return parse_rinex_obs_file_cached(
        rinex_file_path, hash_of_file_content(rinex_file_path)
    )
