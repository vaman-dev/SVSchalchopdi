# Reusable AI Lab Workspace

This repository contains standalone educational experiments for programming,
artificial intelligence, machine learning, robotics, and automation.

Each experiment is kept in a single Python file at the project root so it can
be run locally or pasted into Google Colab. The repository currently preserves
Experiments 02, 04, 05, and 06 and uses Experiment 03 as the active experiment.

## Current experiment

`experiment_03_activation_functions.py` compares ReLU, Sigmoid, Tanh, and
Softmax. It performs the complete workflow:

- implements all four activation functions with NumPy;
- uses numerically stable Sigmoid and Softmax calculations;
- applies each function to the same sample values;
- verifies output ranges, special values, and Softmax probability sums;
- tests extreme inputs and batched Softmax calculations;
- displays activation curves and sample class probabilities;
- measures and plots illustrative NumPy execution time; and
- prints suitable applications, advantages, and limitations.

## Requirements

- Python 3.10 or newer (Python 3.11 is recommended)
- Internet access while installing packages and the first time MNIST is
  downloaded for Experiments 02, 04, or 05
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
NumPy, Matplotlib, Pandas, Scikit-learn, and their required dependencies. Using
the environment's Python executable ensures packages are not installed into
global Python.

If Python 3.11 is unavailable, create the environment with the installed
Python launcher instead:

```powershell
py -m venv env
```

## Run locally

```powershell
env\Scripts\python.exe experiment_02_ann.py
env\Scripts\python.exe experiment_03_activation_functions.py
env\Scripts\python.exe experiment_04_dnn.py
env\Scripts\python.exe experiment_05_cnn_features.py
env\Scripts\python.exe Experiment_06_ethics.py
```

The first MNIST run may take longer while TensorFlow downloads its dataset.
Experiment 03 requires no dataset download. It prints the transformed values,
Softmax sum, measured timing, applications, and limitations, and opens the
activation-behavior and timing figures.

## Run in Google Colab

1. Open a new Google Colab notebook.
2. Paste the contents of `experiment_03_activation_functions.py` into one code
   cell.
3. Run the cell.

If an Experiment 03 import is missing, run this in a preceding cell:

```python
%pip install numpy matplotlib
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
|-- experiment_03_activation_functions.py  Activation comparison experiment
|-- experiment_04_dnn.py        Standalone MNIST DNN experiment
|-- experiment_05_cnn_features.py  Standalone CNN feature experiment
`-- Experiment_06_ethics.py     Standalone Responsible AI experiment
```

For future experiments, preserve completed experiment scripts and replace the
experiment-specific specification, plan, architecture, and progress content in
`docs/`.

Because `AGENTS.md` is stored in `docs/` to keep the root clean, tell a new
Codex session to read `docs/AGENTS.md` before starting work.

## Verified local result

Experiment 03 completed locally on Windows using NumPy 2.2.6 and Matplotlib
3.10.9. ReLU, Sigmoid, Tanh, one-dimensional Softmax, batched Softmax, and
extreme-input checks passed. For 100,000 values, the observed best average
times per call were ReLU `0.0593 ms`, Sigmoid `5.3819 ms`, Tanh `0.8095 ms`,
and Softmax `1.5645 ms`. Timing varies with the environment, and Softmax does
additional vector normalization work. Both plotting paths executed in
headless validation; direct Google Colab execution remains unverified.
