# Complex Satellite Sequence Assembler

Complex satellites are tandem repeats with a large unit size and a some sequence diversity among units.

Our objective is a *de novo* assembler for any type of complex satellite.

Current implementaion supports only:
- Explorative execution e.g. via Jupyter Notebook (i.e. no integrated command)
- PacBio CCS (a.k.a. HiFi reads)

Current implementation requires:
- Python3.7 (not 3.8)
- Unpublished software named Consed (we offer alternative codes, though)

## How to install

1. Clone the repository:

```bash
$ git clone --recursive https://github.com/yoshihikosuzuki/complex-sat-asm
```

2. Create a virtual environment with Python3.7:

```bash
$ virtualenv -p python3.7 complex-sat-asm
$ source complex-sat-asm/bin/activate
```

3. Install codes with dependencies:

```bash
$ python setup.py install
```

## How to use

