# Lab 11 - Labeling and Augmenting a Detection Dataset

Run `lab11_labeling_augmentation.ipynb` from top to bottom.

Keep all files in this folder together. In VS Code, install the packages with
`python -m pip install -r requirements.txt`, then select that same Python
environment as the notebook kernel.

## GitHub Actions execution

The automated execution configuration is stored at
[`.github/workflows/lab11.yml`](../../../.github/workflows/lab11.yml)
at the repository root (GitHub Actions requires workflow files to live there).

Open **Actions > Lab 11** to inspect runs. The workflow installs dependencies,
executes the notebook and uploads the executed notebook as an artifact when
available. This does **not** imply that any run has passed successfully.

## Manual annotation requirement

In notebook step 2, annotate three pedestrian images using `labeler.html`
and put their YOLO `.txt` labels in `work/manual_labels/`.
Without these files, the notebook's quality-check cell reports missing labels;
it does not count as completed manual annotation. The full detection dataset
in step 4 is generated separately from reference instance masks.
