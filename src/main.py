from typing import Dict
import argparse
import toml
from BITS.seq.io import fasta_to_db, save_fasta
from BITS.util.io import load_pickle, save_pickle
from BITS.util.scheduler import Scheduler
from .datander import DatanderRunner
from .datruf import (DatrufRunner,
                     load_qv,
                     filter_reads)
from .overlapper import (UnsyncReadsOverlapper,
                         ReadSynchronizer,
                         SplitMergeDpmmOverlapper,
                         SyncReadsOverlapper,
                         filter_overlaps,
                         best_overlaps_per_pair,
                         remove_contained_reads,
                         adaptive_filter_overlaps)
"""
from .layouter.graph import (overlaps_to_string_graph,
                             reduce_graph)
from .layouter.contig import graphs_to_contigs
"""


# Predefined sets of tasks
EXTRACT = {"fasta_to_db",
           "datander",
           "datruf",
           "load_qv"}
FILTER = {"filter_reads"}
ASSEMBLE = {"unsync_overlap",
            "filter_unsync",
            "sync",
            "smdc",
            "sync_overlap"
            "filter_sync",
            "contig"}
ALL = EXTRACT | FILTER | ASSEMBLE


def main():
    config = parse_args()
    tasks = (EXTRACT if config["task"] == "extract"
             else FILTER if config["task"] == "filter"
             else ASSEMBLE if config["task"] == "assemble"
             else ALL if config["task"] == "all"
             else {config["task"]})
    scheduler = (Scheduler(**config["job_scheduler"]["args"])
                 if config["job_scheduler"]["use_scheduler"]
                 else None)
    # Run specified tasks
    if "fasta_to_db" in tasks and config["extract"]["from_fasta"]:
        fasta_to_db(db_prefix=config["extract"]["db_prefix"],
                    db_type=config["extract"]["db_suffix"],
                    **config["extract"]["fasta_to_db"])
    if "datander" in tasks:
        # Detect tandem repeats
        DatanderRunner(db_prefix=config["extract"]["db_prefix"],
                       db_suffix=config["extract"]["db_suffix"],
                       scheduler=scheduler,
                       **config["extract"]["tr_detection"]).run()
    if "datruf" in tasks:
        # Detect tandem repeat units
        DatrufRunner(db_fname=f"{config['extract']['db_prefix']}.db",
                     las_fname=f"TAN.{config['extract']['db_prefix']}.las",
                     scheduler=scheduler,
                     verbose=config["verbose"],
                     **config["extract"]["unit_detection"]).run()
    if "load_qv" in tasks:
        # Load QV data
        load_qv(reads_fname=config["extract"]["unit_detection"]["out_fname"],
                **config["extract"]["load_qv"])
    # TODO: Visualize unit length distribution (output an html file?)
    if "filter_reads" in tasks:
        # Filter reads having units you want to assemble
        filter_reads(reads_fname=config["extract"]["unit_detection"]["out_fname"],
                     **config["filter"])
    if "unsync_overlap" in tasks:
        # Compute overlaps between unsynchronized TR reads
        UnsyncReadsOverlapper(reads_fname=config["assemble"]["reads_fname"],
                              scheduler=scheduler,
                              verbose=config["verbose"],
                              **config["assemble"]["unsync_overlap"]).run()
    if "filter_unsync" in tasks:
        # Adaptively filter unsynchronized overlaps
        overlaps = load_pickle(
            config["assemble"]["unsync_overlap"]["out_fname"])
        filtered_overlaps = adaptive_filter_overlaps(
            overlaps=overlaps,
            **config["assemble"]["unsync_filter"]["adaptive_args"])
        filtered_overlaps = best_overlaps_per_pair(
            filtered_overlaps,
            by=config["assemble"]["unsync_filter"]["best_overlap_by"])
        if config["assemble"]["unsync_filter"]["contained_removal"]:
            filtered_overlaps = remove_contained_reads(filtered_overlaps)
        save_pickle(filtered_overlaps,
                    config["assemble"]["unsync_filter"]["out_fname"])
    if "sync" in tasks:
        # Syncronize TR reads
        ReadSynchronizer(
            reads_fname=config["assemble"]["reads_fname"],
            overlaps_fname=config["assemble"]["unsync_filter"]["out_fname"],
            scheduler=scheduler,
            verbose=config["verbose"],
            **config["assemble"]["unit_sync"]).run()
    if "smdc" in tasks:
        # Correct errors in units via clustering
        SplitMergeDpmmOverlapper(
            sync_reads_fname=config["assemble"]["unit_sync"]["out_fname"],
            scheduler=scheduler,
            verbose=config["verbose"],
            **config["assemble"]["unit_clustering"]).run()
    if "sync_overlap" in tasks:
        # Compute overlaps between synchronized TR reads with corrected units
        SyncReadsOverlapper(
            sync_reads_fname=config["assemble"]["unit_clustering"]["out_fname"],
            scheduler=scheduler,
            verbose=config["verbose"],
            **config["assemble"]["sync_overlap"]).run()
    if "filter_sync" in tasks:
        # Adaptively filter synchronized overlaps
        overlaps = load_pickle(config["assemble"]["sync_overlap"]["out_fname"])
        filtered_overlaps = filter_overlaps(
            overlaps,
            max_diff=config["assemble"]["sync_filter"]["max_diff"],
            min_ovlp_len=config["assemble"]["sync_filter"]["min_ovlp_len"])
        filtered_overlaps = best_overlaps_per_pair(
            filtered_overlaps,
            by=config["assemble"]["sync_filter"]["best_overlap_by"])
        filtered_overlaps = adaptive_filter_overlaps(
            overlaps=filtered_overlaps,
            filter_by_diff=False,
            **config["assemble"]["sync_filter"]["adaptive_args"])
        save_pickle(filtered_overlaps,
                    config["assemble"]["sync_filter"]["out_fname"])
    """
    if "contig" in tasks:
        # Construct a string graph
        sg = overlaps_to_string_graph(filtered_overlaps)
        sg_ccs = reduce_graph(sg)
        # Generate contigs
        reads = load_pickle(config["assemble"]["reads_fname"])
        reads_by_id = {read.id: read for read in reads}
        contigs = graphs_to_contigs(sg_ccs,
                                    filtered_overlaps,
                                    reads_by_id,
                                    n_core=config["assemble"]["layout"]["n_core"])
        save_fasta(contigs, config["assemble"]["layout"]["out_fname"])
    """


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
    assert args.task_name in {"extract", "filter", "assemble", "all"} | ALL, \
        f"Invalid task name: {args.task_name}"
    config = toml.load(args.config_fname)
    config["task"] = args.task_name
    return config


if __name__ == "__main__":
    main()
