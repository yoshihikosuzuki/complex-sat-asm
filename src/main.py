from typing import Dict
import argparse
import toml
from BITS.util.scheduler import Scheduler
from .datander import DatanderRunner
from .datruf import (DatrufRunner,
                     filter_by_tr,
                     filter_by_unit)
from .overlapper import (UnsyncReadsOverlapper,
                         ReadSynchronizer,
                         SplitMergeDpmmOverlapper,
                         SyncReadsOverlapper)
from .layouter import (overlaps_to_graph,
                       reduce_graph,
                       graphs_to_contigs)


def main():
    config = parse_args()
    scheduler = (Scheduler(**config["job_scheduler"]["args"])
                 if config["job_scheduler"]["use_scheduler"]
                 else None)
    if config["task"] in ("extract", "all"):
        # Detect tandem repeats
        DatanderRunner(db_prefix=config["extract"]["db_prefix"],
                       scheduler=scheduler,
                       **config["extract"]["tr_detection"]).run()
        # Detect tandem repeat units
        DatrufRunner(db_fname=f"{config['extract']['db_prefix']}.db",
                     las_fname=f"TAN.{config['extract']['db_prefix']}.las",
                     # TODO: pass arguments for loading QV
                     scheduler=scheduler,
                     **config["extract"]["unit_detection"]).run()
    if config["task"] in ("filter", "all"):
        # Filter reads having units you want to assemble
        pass
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
