# Complex Satellite Sequence Assembly

Complex satellites are tandem repeats with a large (e.g. >50 bp) unit size and often some sequence diversity among units.

Our objective is a _de novo_ assembler for any type of complex satellites with PacBio reads (especially CCS a.k.a. HiFi reads; we developed mainly with a [40x subset](https://obj.umiacs.umd.edu/marbl_publications/hicanu/index.html) of [Drosophila 24 kb dataset](https://www.ncbi.nlm.nih.gov/sra/SRX6957826) used in the [HiCanu paper](https://www.biorxiv.org/content/10.1101/2020.03.14.992248v3)).

## Requirements

- Python 3.7 (not >=3.8)
- [DAZZ_DB](https://github.com/thegenemyers/DAZZ_DB)
- [DALIGNER](https://github.com/thegenemyers/DALIGNER)
- Custom [DAMASKER](https://github.com/yoshihikosuzuki/DAMASKER) (cloned togeher with this repository)
- A python package named `BITS` (go to the `BITS/` directory of this repository and install it; Note that it is old version of [this package](https://github.com/yoshihikosuzuki/bits))
- Unpublished software named Consed for computing a consensus of multiple sequences (we will offer alternative codes; detection of tandem repeats and units does not require it)

## How to install

```bash
(Install DAZZ_DB, DALIGNER)
$ git clone --recursive https://github.com/yoshihikosuzuki/complex-sat-asm
$ cd complex-sat-asm
(Install DAMASKER, BITS(, consed_python))
$ python setup.py install
```

## How to use

```
usage: csa [-h] [-c CONFIG_FNAME] task_name [task_name ...]

CSA: Complex Satellite Assembler

positional arguments:
  task_name             Must be subset of {'extract', 'filter', 'assemble',
                        'all'}.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FNAME, --config_fname CONFIG_FNAME
                        TOML file. [config.toml]
```

You need followings as input:

- A file of PacBio reads (`.fasta` file or `.db` file)
- Config file (`config.toml` by default; template file named `config.template.toml` is put at the root of this repository)
- Task name as a command-line option
  - `extract`: Detect tandem repeats and their units from reads
  - `filter`: Filter reads to be assembled by specifying unit lengths. If you don't have prior information on the unit size of the satellites you wish to assemble, draw and check the distribution of unit lengths (described below) and determine their values before this step.
  - `assemble`: Assemble the filtered reads.
  - `all`: Run all the tasks above.

### Parameters in the config file

[TODO]

### Visualization modules

[TODO]
