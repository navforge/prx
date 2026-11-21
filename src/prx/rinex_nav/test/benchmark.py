import logging

import pandas as pd
from pathlib import Path
from prx.rinex_nav import evaluate as rinex_nav_evaluate
from prx import converters
import cProfile
logger = logging.getLogger(__name__)


def generate_query(n_epochs=1):
    rinex_nav_file = converters.compressed_to_uncompressed(
        Path(__file__).parent / "datasets/BRDC00IGS_R_20220010000_01D_MN.zip"
    )
    df = rinex_nav_evaluate.parse_rinex_nav_file(rinex_nav_file)
    n_ephemerides = df["ephemeris_hash"].nunique()
    rinex_nav_evaluate.cached_load.cache_clear()
    df["constellation"] = df["sv"].str[0]
    sats = []
    # Grab ten (or all, if the constellation has less than ten) satellites of each constellation as an approximation
    # of what an open-sky receiver would see
    for _, constellation_df in df.groupby("constellation"):
        constellation_sats = constellation_df["sv"].unique()
        sats.extend(constellation_sats[0 : min(10, len(constellation_sats))])
    query_template = pd.DataFrame(
        {
            "sv": sats,
            "signal": "C1C",
            "query_time_isagpst": pd.Timestamp("2022-01-01T01:10:00.000000000"),
        }
    )
    query = query_template.copy()
    for i in range(1, n_epochs):
        next_query = query_template.copy()
        next_query["query_time_isagpst"] += pd.Timedelta(seconds=i)
        query = pd.concat((query, next_query), axis=0)
    return query, n_ephemerides


def benchmark(query, rinex_nav_file):
    return rinex_nav_evaluate.compute_parallel(rinex_nav_file, query)


if __name__ == "__main__":
    rinex_nav_file = converters.compressed_to_uncompressed(
        Path(__file__).parent / "datasets/BRDC00IGS_R_20220010000_01D_MN.zip"
    )
    query, n_ephemerides = generate_query(100)
    print(
        f"Profiling ephemeris evaluation with {len(query.sv.unique())} satellites and"
        f" {len(query.query_time_isagpst.unique())} epochs"
    )
    p = cProfile.Profile()
    p.enable()
    benchmark(query, rinex_nav_file)
    p.disable()
    stats_file = Path(__file__).parent / "benchmark_1.prof"
    p.dump_stats(stats_file)
    df = (
        pd.DataFrame(
            p.getstats(),
            columns=["func", "ncalls", "ccalls", "tottime", "cumtime", "callers"],
        )
        .sort_values(by="tottime", ascending=False)
        .reset_index(drop=True)
    )[["func", "tottime"]]
    df["function"] = df["func"].apply(lambda x: getattr(x, "co_name", None))
    df["file"] = df["func"].apply(lambda x: getattr(x, "co_filename", None))
    df = df[df["function"].notnull() & df["file"].str.contains(r"prx/src/prx")]
    not_interesting = ["timeit_wrapper", "<genexpr>"]
    df = df[~df["function"].isin(not_interesting)]
    df = df.drop(columns=["func"])
    t_parse_s = df[df['function']=="parse_rinex_nav_file"]["tottime"].to_list()[0]
    t_eval_s = df[df['function']=="compute_parallel"]["tottime"].to_list()[0] - t_parse_s
    print(
        f"Approximate nav file parsing speed: {n_ephemerides/t_parse_s} ephemerides/s"
    )
    print(
        f"Approximate ephemeris evaluation speed: {len(query.query_time_isagpst.unique()) / t_eval_s} epochs/s"
    )
    print(f"To inspect profiling results, call \n snakeviz {stats_file}")
