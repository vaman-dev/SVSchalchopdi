# Reusable AI Lab Workspace

This repository contains standalone educational experiments for programming,
artificial intelligence, machine learning, robotics, and automation.

Each experiment is kept in a single Python file at the project root so it can
be run locally or pasted into Google Colab. The repository currently preserves
Experiment 02 and uses Experiment 04 as the active experiment.

## Current experiment

`experiment_04_dnn.py` implements a fully connected deep neural network for
MNIST image classification. It performs the complete workflow:

- downloads and loads the MNIST dataset;
- normalizes image pixels to the range `[0, 1]`;
- creates separate training and validation partitions;
- builds a `Flatten -> Dense(128) -> Dense(64) -> Dense(10)` DNN;
- trains for five epochs;
- evaluates once on the held-out test set;
- predicts one test digit; and
- displays training and validation accuracy/loss graphs.

## Requirements

- Python 3.10 or newer (Python 3.11 is recommended)
- Internet access the first time MNIST is downloaded
- The packages listed in `requirements.txt`

## Install all required packages

Open PowerShell in the repository root. Create the virtual environment once if
`env/` does not already exist:

```powershell
py -3.11 -m venv env
```

Install every package from `requirements.txt` into that environment:

```powershell
env\Scripts\python.exe -m pip install --upgrade pip
env\Scripts\python.exe -m pip install -r requirements.txt
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
```

The first run may take longer while TensorFlow downloads MNIST. The program
prints dataset shapes, the model summary, training progress, test loss, test
accuracy, and a sample prediction. It also opens the training-history plots.

## Run in Google Colab

1. Open a new Google Colab notebook.
2. Paste the contents of `experiment_04_dnn.py` into one code cell.
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
|-- AGENTS.md                   Permanent workspace instructions
|-- EXPERIMENT_SPEC.md          Requirements for the active experiment
|-- PLAN.md                     Milestones for the active experiment
|-- ARCHITECTURE.md             Program design and data flow
|-- PROGRESS.md                 Verified results, blockers, and next action
|-- experiment_02_ann.py        Standalone MNIST ANN experiment
|-- experiment_04_dnn.py        Standalone MNIST DNN experiment
|-- requirements.txt            Python dependencies
`-- README.md                   Setup and usage instructions
```

For future experiments, preserve completed experiment scripts and replace the
experiment-specific specification, plan, architecture, and progress content.

## Verified local result

Experiment 04 completed a five-epoch CPU run on Windows with TensorFlow 2.21.0.
That run produced a test accuracy of `0.9682` and test loss of `0.1029`.
Results can vary slightly with hardware and library versions.
