# Rank Estimation Pipeline

A pipeline for computing the rank of Jacobian matrices from quantum states.

## Usage

```bash
python pipeline.py --N <size> --psi <state> --impl <implementation>
```

### Arguments

- `--N`: System size (default: 8)
- `--psi`: Psi state function - `product` or `random` (default: random)
- `--impl`: Implementation - `opt2` or `opt3` (default: opt2)

### Examples

```bash
# Basic usage
python pipeline.py --N 8 --psi random --impl opt2
python pipeline.py --N 10 --psi product --impl opt3

# For long runs (prevents Mac sleep)
caffeinate -i python pipeline.py --N 16 --psi random --impl opt2
```

## Output

- Progress updates every 30 seconds with ETA
- Spectrum plot displayed and saved to `images/N_<size>_<psi>/` folder
- Final rank printed at completion

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
``` 