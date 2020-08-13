from typing import Dict
import argparse
import toml
from BITS.seq.io import fasta_to_db
from BITS.util.scheduler import Scheduler
from .datander import DatanderRunner
from .datruf import (DatrufRunner,
                     load_qv,
                     filter_reads)
"""
from .overlapper import (UnsyncReadsOverlapper,
                         ReadSynchronizer,
                         SplitMergeDpmmOverlapper,
                         SyncReadsOverlapper)
from .layouter import (overlaps_to_graph,
                       reduce_graph,
                       graphs_to_contigs)
"""


def main():
    config = parse_args()
    scheduler = (Scheduler(**config["job_scheduler"]["args"])
                 if config["job_scheduler"]["use_scheduler"]
                 else None)
    if config["task"] in ("extract", "all"):
        if config["extract"]["from_fasta"]:
            fasta_to_db(db_prefix=config["extract"]["db_prefix"],
                        db_type="db",
                        **config["extract"]["fasta_to_db"])
        # Detect tandem repeats
        DatanderRunner(db_prefix=config["extract"]["db_prefix"],
                       scheduler=scheduler,
                       **config["extract"]["tr_detection"]).run()
        # Detect tandem repeat units
        DatrufRunner(db_fname=f"{config['extract']['db_prefix']}.db",
                     las_fname=f"TAN.{config['extract']['db_prefix']}.las",
                     scheduler=scheduler,
                     out_fname=config["extract"]["out_fname"],
                     verbose=config["verbose"],
                     **config["extract"]["unit_detection"]).run()
        # Load QV data
        load_qv(reads_fname=config["extract"]["out_fname"],
                **config["extract"]["load_qv"])
        # TODO: Visualize unit length distribution (output an html file?)
    if config["task"] in ("filter", "all"):
        # Filter reads having units you want to assemble
        filter_reads(reads_fname=config["extract"]["out_fname"],
                     **config["filter"])
    if config["task"] in ("assemble", "all"):
        pass


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        description="CSA: Complex Satellite Assembler")
    parser.add_argument(
        "task_name",
        type=str,
        help="Must be one of {'extract', 'filter', 'assemble', 'all'}.")
    parser.add_argument(
        "-c",
        "--config_fname",
        type=str,
        default="config.toml",
        help="TOML file. [config.toml]")
    args = parser.parse_args()
    assert args.task_name in ("extract", "filter", "assemble", "all"), \
        f"Invalid task name: {args.task_name}"
    config = toml.load(args.config_fname)
    config["task"] = args.task_name
    return config


if __name__ == "__main__":
    main()
