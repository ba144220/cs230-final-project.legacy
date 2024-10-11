# cs230-final-project

## Environment

For the following instructions, `mamba` and `conda` are interchangeable. `mamba` is recommended.

1. Create the environment:
```bash
mamba env create -f environment.yaml
```

2. Activate the environment:
```bash
mamba activate cs230
```
If you want to deactivate the environment, run `mamba deactivate`.

3. Install new packages:
Add the package to `environment.yaml` and run
```bash
mamba env update -f environment.yaml
```
If you use `pip` to install packages, add them to `environment.yaml` under `pip` section.

**YOU SHOULD NOT INSTALL NEW PACKAGES DIRECTLY THROUGH `conda`, `mamba`, or `pip`.**

## Environment Variables File

Create a `.env` file based on `.env.example` and put your Hugging Face token in it.
