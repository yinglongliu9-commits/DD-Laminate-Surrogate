# DD Laminate Surrogate Models

Training code for parameter-driven prediction of impact damage images and ultimate compressive loads in double-double (DD) composite laminates.

## Models

- Damage model: a FiLM-conditioned decoder predicts paired RGB contours of matrix tensile damage (SDV9) and matrix compressive damage (SDV10) from material, layup and ply-specific inputs.
- Load model: an independently trained multilayer perceptron predicts ultimate CAI and CBI loads from material and layup inputs.

The image model reconstructs rendered contours, not integration-point damage variables. The internal task name `strength` refers to compressive **loads**, not stresses.

This release contains training code only. Datasets, trained weights, simulation files and generated results are not included. Training requires data arranged as described below.

## Requirements

Python 3.10 or later is recommended. Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your CUDA environment. The script selects CUDA automatically when available and otherwise uses the CPU.

## Data Layout

Place the dataset beside `train_composite.py`:

```text
dataset/
  material_parameters.csv
  strength/
    M1_CAI.csv
    M1_CBI.csv
    ... corresponding files for each material
  PNG/
    M1/
      SDV9/
        0_0_1_0.png
        ...
      SDV10/
        0_0_1_0.png
        ...
    ... corresponding directories for each material
```

- `material_parameters.csv`: `material_id` identifies the material. The original input schema is `material_id, source_inp, material1_density, um01, ..., um27`. The `um` columns follow the material-parameter ordering used to construct the dataset. `source_inp` is metadata; remaining columns must be numeric, except the additional excluded metadata columns named in `material_numeric_columns`.
- Load CSV files: columns `ID` and `Max_RT`. `Max_RT` contains the ultimate compressive load in consistent units. IDs follow `ID = b + 10*a + 1`.
- Layup indices `a` and `b` run from 0 to 9, corresponding to angles `10*a` and `10*b` in degrees. The eight-ply sequence is `[+10a, -10a, +10b, -10b, -10b, +10b, -10a, +10a]`.
- Image names follow `{a}_{b}_{layer}_{angle}.png`, with layers numbered 1 through 8 and signed ply angles in degrees. The script constructs integer angle strings, including negative values.
- Both damage images are converted to RGB and resized to 256 x 256 pixels.

The load task only needs the material and load CSV files; the damage task only needs the material CSV and images.

## Training

Train the damage-image model:

```sh
python train_composite.py --task damage
```

Train the CAI/CBI load model:

```sh
python train_composite.py --task strength
```

Train both models independently in sequence:

```sh
python train_composite.py --task all
```

View configurable options:

```sh
python train_composite.py --help
```

Default settings are defined in `DEFAULT_CONFIG`; command-line arguments override them. The damage model uses 256 x 256 outputs; the retained `--image-size` argument does not change that fixed resolution.

## Outputs and Splits

Outputs are written under `runs/composite/damage` and `runs/composite/strength`, including best-validation checkpoints, training histories, metadata and configured visualizations.

Each task uses a random 80%/10%/10% training/validation/test split with seed 42 by default. Input normalization is fitted on the training subset; load targets are standardized using training targets. The image split operates at the individual ply-sample level, not by held-out layup or material. These splits do not establish generalization to unseen layups or unseen materials. Saved metadata records the split indices.

Use a new `--output-dir` for a separate experiment: existing training histories and checkpoints in the same output location can be overwritten.

## Scope

This repository preserves the original training implementation. It does not include damage-transfer routines, separate inference scripts, SHAP post-processing or manuscript plotting utilities.
