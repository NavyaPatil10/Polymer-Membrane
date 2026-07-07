from flask import Flask, render_template, request
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

app = Flask(__name__)

# ==========================================
# Load Trained Models
# ==========================================

model_co2 = joblib.load("final_xgb_CO2.pkl")
model_n2 = joblib.load("final_xgb_N2.pkl")
model_co2_n2 = joblib.load("final_xgb_CO2_N2.pkl")


# ==========================================
# Fingerprint Function
# ==========================================

def smiles_to_ecfp(smiles, radius=2, nBits=2048):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=nBits
    )

    arr = np.zeros((nBits,), dtype=np.float32)

    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr.reshape(1, -1)


# ==========================================
# Home Page
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        smiles = request.form["smiles"].strip()

        fingerprint = smiles_to_ecfp(smiles)

        if fingerprint is None:

            error = "Invalid SMILES! Please enter a valid SMILES."

        else:

            # Predict log10 values
            log_co2 = float(model_co2.predict(fingerprint)[0])
            log_n2 = float(model_n2.predict(fingerprint)[0])
            log_co2_n2 = float(model_co2_n2.predict(fingerprint)[0])

            # Convert back to original values
            co2 = 10 ** log_co2
            n2 = 10 ** log_n2
            co2_n2 = 10 ** log_co2_n2

            prediction = {
                "CO2": round(co2, 3),
                "N2": round(n2, 3),
                "CO2_N2": round(co2_n2, 3)
            }

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


# ==========================================
# Run Flask
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
