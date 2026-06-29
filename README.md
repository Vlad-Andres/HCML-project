# Human-Centered Machine Learning Project

We analyzes the Open University Learning Analytics Dataset (OULAD) to evaluate fairness and explainability across demographic and different subgroups.

## How to run

We use [uv](https://github.com/astral-sh/uv) to manage project dependencies and virtual environments.

### 1. Install `uv` (if not already installed)
```bash
brew install uv
```

### 2. Set up the environment and sync dependencies
Run the following command in the project root:
```bash
uv sync
```
Now you can run the notebook.

## Dataset

The dataset contains these tables that we utilized:

- `courses.csv`
- `assessments.csv`
- `studentAssessment.csv`
- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`
- `vle.csv`

> Our report can be found in the root of the project as `HCML_Report.pdf`.

## Project Structure

- `src/config.py` — configuration constants, dataset keys, and required file names
- `src/data_loader.py` — data loading and OULAD file resolution helpers
- `src/feature_engineering.py` — target creation, VLE/assessment aggregation, registration features, and stage dataset construction
- `src/modeling.py` — preprocessing, train/test splitting, model building, training, and evaluation
- `src/explainability.py` — SHAP analysis and feature family importance
- `src/fairness.py` — subgroup fairness metrics and gap summaries

