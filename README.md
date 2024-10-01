# Chest X-Ray Case-Based Explanations

## Structure
```
├── config.yaml
├── README.md
├── src
│   ├── data
│   ├── federated             - FL Client and Server (unused)
│   ├── forward_catalogue.py  - Obtain explanation ranking for images chosen from a limited catalogue
│   ├── forward_grad.py       - Obtain grad cam for test images
│   ├── forward.py            - Obtain the best explanations from the entire MIMIC dataset
│   ├── models                - Model definition
│   │   └── ...
│   ├── modules               - Pytorch Lightning module definition
│   │   └── ...
│   ├── sim.py                - Run Federated Learning simulation
│   ├── test_centralized.py   - Test model in a centralized setting
│   ├── test.py               - Test model for MIMIC-CXR-JPG (legacy)
│   ├── train_centralized.py  - Train model in a centralized setting 
│   ├── train.py              - Train model for MIMIC-CXR-JPG (legacy)
│   └── train_synthetic.py    - Train model with a synthetic dataset
```

## Setup

Using venv or your virtual environment of choice,
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optionally, for development, pre-commit hooks can be installed using:
```sh
pip install pre-commit
pre-commit install
```
