---
title: Getting started
nav_order: 2
---

# Getting started
{: .no_toc }

1. TOC
{:toc}

## Install

You need Python 3.11 or 3.12 (TensorFlow does not yet support newer versions). [uv](https://docs.astral.sh/uv/) is the easiest way to get a matching environment:

```bash
git clone https://github.com/AI-in-Transportation-Lab/ai-in-he.git
cd ai-in-he
uv sync                                   # creates .venv and installs dependencies
uv run python scripts/run_all.py          # run every case study
```

Using pip instead:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
```

## Running a subset

```bash
python scripts/run_all.py --list              # show what would run
python scripts/run_all.py --chapter 6         # one chapter
python scripts/run_all.py -c 3 -k 2           # chapter 3, case study 2
make ch08                                     # same thing via make
```

The runner executes each script with `outputs/chNN/` as its working directory, so figures and files the listings write with relative paths land there and never in the source tree. You can also run a script directly; output then goes to your current directory.

## Chapter 4, Case Study 2 (object detection)

This case study needs a YOLO11n ONNX model and a sample street image, which are not stored in git:

```bash
python scripts/fetch_assets.py            # downloads into data/
python scripts/run_all.py -c 4 -k 2
```

## Repository layout

```
ai-in-he/
├── chapters/chNN_<topic>/cs<N>_<topic>.py   # one script per case study
├── scripts/run_all.py, fetch_assets.py
├── docs/figures/                            # reference figures from the book
├── data/                                    # downloaded assets (git-ignored)
└── outputs/                                 # generated figures (git-ignored)
```

## Reproducibility notes

- All data are synthetic and generated with fixed seeds. Numbers can still differ slightly across library versions and platforms, in particular for the TensorFlow models in Chapters 2 and 3. `uv.lock` pins the versions used to check these scripts.
- The synthetic data illustrate methods. They are not agency data, and results should not be read as findings about real roadways.
- The scripts are the book's listings verbatim; only a short docstring header was added to each.

## Previewing this site locally

```bash
bundle install
bundle exec jekyll serve    # http://localhost:4000/ai-in-he/
```
