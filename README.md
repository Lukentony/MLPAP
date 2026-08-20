# Machine Learning Predictive Analysis Project (MLPAP)

A Python pipeline skeleton for tabular predictive analysis with scikit-learn: load a CSV
dataset, prepare features/target, preprocess, split, train and evaluate a regression model,
and save the trained model.

## What's here

- `src/config.py` — central configuration: data directory, model output path, train/test
  split, dataset filename, target column, date/categorical columns.
- `src/data/data_loader.py` — loads a CSV from the configured directory, derives day/month/year
  columns from date columns, and splits data into train/test sets.
- `src/data/data_processor.py` — imputes missing values, label-encodes categorical columns, and
  standard-scales numeric columns.
- `src/models/model.py` — wraps a scikit-learn `RandomForestRegressor` with train/predict/save/load
  methods.
- `src/models/model_trainer.py` — trains the model and reports Mean Squared Error and R² on a
  test split.
- `src/utils/logger.py` — configures a console logger.
- `src/utils/evaluation.py` — present but empty.
- `src/main.py` — orchestrates the pipeline: load → prepare features/target → preprocess →
  split → train/evaluate → save.

The default configuration in `src/config.py` points at a dataset file named
`covid19_italy_province.csv` with target column `TotalPositiveCases`. No dataset file is
included in this repository.

## Dependencies

Listed in `src/requirements.txt`: pandas, numpy, scikit-learn, joblib.

## Status

As currently committed, the code is not runnable as-is:
- `src/main.py` imports modules by paths that don't match the actual package layout (e.g.
  `src.data_loader`, `src.predictive_model`, `src.model_trainer` instead of
  `src.data.data_loader`, `src.models.model`, `src.models.model_trainer`).
- `src/models/model.py` and `src/data/data_processor.py` reference `Config` without importing it.
- `src/utils/evaluation.py` is an empty file.

No LICENSE file is present in this repository.
