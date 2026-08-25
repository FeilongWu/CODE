# When Clinical Notes Are Incomplete: Fusing Clinical Notes and Time Series Using Diffusion-based Models
## (EMNLP 2026)
This implementation contains CODE for classification tasks and its validation using synthetic notes. CODE is a diffusion-based model to recover complete semantics of a [CLS] embedding of a incomplete note.
## Data Preparation
For MIMIC-III, please follow the instructions under "/data". To prepare synthetic notes for validation, run "create_incomplete.py" under "synthetic_report".

## Train
Use the following command to train and test CODE.
'''console
python main.py
'''
