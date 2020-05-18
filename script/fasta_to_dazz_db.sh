#!/bin/bash

N_ARGS=3
if [ $# -ne ${N_ARGS} ]; then
    echo '
Usage: ./fasta_to_dazz_db.sh <in_fasta> <out_db_prefix> <db_block_size>

Create a DAZZ_DB file from a fasta file.

Options
-------

in_fasta [str]
    Name of an input fasta file.

out_db_prefix [str]
    Prefix name of an output DAZZ_DB file.

db_block_size [int]
    Size of a block (in MB) of the output DAZZ_DB file.
    Used for the `DBsplit` command.
'
    exit 1
fi

IN_FASTA=$1
OUT_DB_PREFIX=$2
DB_BLOCK_SIZE=$3

fasta2DB ${OUT_DB_PREFIX} ${IN_FASTA}
DBsplit -s${DB_BLOCK_SIZE} ${OUT_DB_PREFIX}
