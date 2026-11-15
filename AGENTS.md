# AGENTS.md — Reusable Codex Lab Workspace

## Purpose
This repository is used for multiple programming, AI,
ML, robotics, and automation experiments.

Each experiment must be simple, independently runnable,
educational, and easy to explain.

## Mandatory workflow
1. Read EXPERIMENT_SPEC.md, PLAN.md, ARCHITECTURE.md,
   and PROGRESS.md before implementing anything.
2. For a new experiment, first analyze its requirements.
3. Create or update PLAN.md with ordered milestones.
4. Create or update ARCHITECTURE.md with the design.
5. Implement one milestone at a time.
6. Use specialist subagents only when useful.
7. Apply reusable skills when they match the task.
8. Verify the actual code and record results.
9. Update PROGRESS.md before compacting context.

## Code structure
- Prefer one standalone Python file per experiment.
- Naming: experiment_02_ann.py, experiment_03_name.py.
- Keep runnable experiment files in the project root.
- Do not create src/, packages, notebooks/, or
  unnecessary abstractions for simple experiments.
- Code must work locally and by pasting into Colab.
- Avoid relative imports and local file dependencies.
- Use clear comments for important operations.
- Prefer straightforward code over complex design patterns.
- Never overwrite previous experiment scripts.


## Python Environment Policy
- Use the project's existing env/ virtual environment.
- Never install packages into global Python.
- Use env/Scripts/python.exe for Windows execution.
- Install dependencies only when required.
- Verify the package installation location.
- Do not reinstall dependencies after successful tests.
- Never assume Codex sandbox changes are temporary.
- Do not delete or recreate env/ without permission.

## Planning requirements
Before implementation, document:
- Problem statement and objectives
- Required libraries and dataset
- Step-by-step algorithm
- Proposed architecture
- Milestones and acceptance criteria
- Expected outputs and verification strategy

## Delegation
- coder: implement isolated tasks.
- reviewer: independently inspect correctness.
- Do not delegate trivial work unnecessarily.
- Avoid concurrent edits to the same file.

## Verification
- Check Python syntax.
- Run applicable tests or lightweight assertions.
- Run the complete experiment when dependencies
  and required resources are available.
- Never claim success or accuracy without execution.
- Clearly identify anything that remains unverified.

## Persistent context
PLAN.md: current milestones and task status.
ARCHITECTURE.md: technical design.
PROGRESS.md: verified outcomes and next action.
EXPERIMENT_SPEC.md: current experiment requirements.

Before /compact, update PROGRESS.md with:
- Completed milestones
- Changed files
- Commands actually executed
- Observed outputs
- Known errors
- Next action

## Scope control
- Do not add frameworks without justification.
- Do not install unnecessary libraries.
- Preserve previous experiment code.
- Do not invent results or execution logs.
- Ask before making destructive changes.
