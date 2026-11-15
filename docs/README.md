# Reusable AI Lab Workspace

This repository contains standalone educational experiments for programming,
artificial intelligence, machine learning, robotics, and automation.

Each experiment is kept in a single Python file at the project root so it can
be run locally or pasted into Google Colab. The repository currently preserves
Experiments 02 and 04 and uses Experiment 05 as the active experiment.

## Current experiment

`experiment_05_cnn_features.py` implements CNN feature extraction and
visualization using MNIST. It performs the complete workflow:

- downloads and loads the MNIST dataset;
- normalizes image pixels to the range `[0, 1]`;
- creates separate training and validation partitions;
- builds and trains a two-convolution CNN;
- trains for five epochs;
- evaluates once on the held-out test set;
- predicts one test digit;
- extracts intermediate activations; and
- displays training history, learned filters, and feature maps.

## Requirements

- Python 3.10 or newer (Python 3.11 is recommended)
- Internet access the first time MNIST is downloaded
- The packages listed in `config/requirements.txt`

## Install all required packages

Open PowerShell in the repository root. Create the virtual environment once if
`env/` does not already exist:

```powershell
py -3.11 -m venv env
```

Install every package from `config/requirements.txt` into that environment:

```powershell
env\Scripts\python.exe -m pip install --upgrade pip
env\Scripts\python.exe -m pip install -r config\requirements.txt
```

The second command is the main installation command. It installs TensorFlow,
NumPy, Matplotlib, and their required dependencies. Using the environment's
Python executable ensures packages are not installed into global Python.

If Python 3.11 is unavailable, create the environment with the installed
Python launcher instead:

```powershell
py -m venv env
```

## Run locally

```powershell
env\Scripts\python.exe experiment_02_ann.py
env\Scripts\python.exe experiment_04_dnn.py
env\Scripts\python.exe experiment_05_cnn_features.py
```

The first run may take longer while TensorFlow downloads MNIST. The program
prints dataset shapes, the model summary, training progress, test loss, test
accuracy, and a sample prediction. Experiment 05 also opens learned-filter and
intermediate feature-map visualizations.

## Run in Google Colab

1. Open a new Google Colab notebook.
2. Paste the contents of `experiment_05_cnn_features.py` into one code cell.
3. Run the cell.

TensorFlow, NumPy, and Matplotlib are normally available in Colab. If an import
is missing, run this in a preceding cell:

```python
%pip install tensorflow numpy matplotlib
```

## Project structure

```text
.
|-- .agents/skills/             Reusable implementation and validation skills
|-- .codex/agents/              Reusable coder and reviewer definitions
|-- config/
|   `-- requirements.txt        Python dependencies
|-- docs/
|   |-- AGENTS.md               Permanent workspace instructions
|   |-- ARCHITECTURE.md         Program design and data flow
|   |-- EXPERIMENT_SPEC.md      Requirements for the active experiment
|   |-- PLAN.md                 Milestones for the active experiment
|   |-- PROGRESS.md             Verified results, blockers, and next action
|   `-- README.md               Setup and usage instructions
|-- experiment_02_ann.py        Standalone MNIST ANN experiment
|-- experiment_04_dnn.py        Standalone MNIST DNN experiment
`-- experiment_05_cnn_features.py  Standalone CNN feature experiment
```

For future experiments, preserve completed experiment scripts and replace the
experiment-specific specification, plan, architecture, and progress content in
`docs/`.

Because `AGENTS.md` is stored in `docs/` to keep the root clean, tell a new
Codex session to read `docs/AGENTS.md` before starting work.

## Verified local result

Experiment 05 completed a five-epoch CPU run on Windows with TensorFlow 2.21.0.
That run produced test accuracy `0.9824` and test loss `0.0544`; all filter and
feature-map plotting paths also executed. Results can vary slightly with
hardware and library versions.
