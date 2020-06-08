# Complex Satellite Sequence Assembly

Complex satellites are tandem repeats with a large (e.g. >100 bp) unit size and a some sequence diversity among units.

Our objective is a _de novo_ assembler for any type of complex satellite.

Current implementaion supports only:

- Explorative execution via e.g. Jupyter Notebook (i.e. no integrated command from shell)
- PacBio CCS a.k.a. HiFi reads (we developed mainly with a [40x subset](https://obj.umiacs.umd.edu/marbl_publications/hicanu/index.html) of [Drosophila 24 kb dataset](https://www.ncbi.nlm.nih.gov/sra/SRX6957826) used in the [HiCanu paper](https://www.biorxiv.org/content/10.1101/2020.03.14.992248v3))

Current implementation requires:

- Python3.7 (not 3.8)
- Unpublished software named Consed (we will offer alternative codes)

## How to install

1. Clone the repository:

```bash
$ git clone --recursive https://github.com/yoshihikosuzuki/complex-sat-asm
$ cd complex-sat-asm
```

2. Create a virtual environment with Python3.7:

```bash
$ python3.7 -m venv .venv
$ source .venv/bin/activate
```

3. Install codes with dependencies:

```bash
$ python setup.py install
```

4. Add the virtual environment to Jupyter as a kernel

```bash
$ bash BITS/script/add_jupyter_kernel.sh .venv CSA
```

## How to use
