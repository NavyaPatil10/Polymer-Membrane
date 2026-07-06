# Polymer-Membrane

## Overview
This project predicts the gas separation performance of polymer membranes using Machine Learning. The model estimates CO₂, O₂, and N₂ permeability along with CO₂/N₂ and CO₂/O₂ selectivity from polymer SMILES representations.

## Features
- CO₂ permeability prediction
- O₂ permeability prediction
- N₂ permeability prediction
- CO₂/N₂ selectivity prediction
- CO₂/O₂ selectivity prediction
- SMILES-based prediction
- Flask web interface

## Technologies Used
- Python
- Flask
- Scikit-learn
- XGBoost
- RDKit
- Pandas
- NumPy

## Dataset
The models were trained using polymer membrane data represented by SMILES strings and Extended Connectivity Fingerprints (ECFP).

## Installation

Clone the repository:

```bash
git clone https://github.com/NavyaPatil10/Polymer-Membrane.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```



## Machine Learning Models
- Random Forest
- XGBoost
- CatBoost

## Author
Navya Patil
M.Tech, Computer Science and Engineering
BMS College of Engineering
