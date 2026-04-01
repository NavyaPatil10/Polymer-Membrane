from flask import Flask, render_template, request
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)

# ===============================
# LOAD MODELS
# ===============================
try:
    model_co2 = joblib.load("final_xgb_CO2.pkl")
    model_n2  = joblib.load("final_xgb_N2.pkl")
    model_o2  = joblib.load("final_xgb_O2.pkl")
    print("✅ Models loaded successfully")
except Exception as e:
    print("❌ Model loading error:", e)
    model_co2 = model_n2 = model_o2 = None


# ===============================
# SMILES → ECFP (FAST VERSION)
# ===============================
def smiles_to_ecfp(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=n_bits
    )

    # Fast conversion
    arr = np.array(fp).reshape(1, -1)

    return arr


# ===============================
# ROUTE
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":

        smiles = request.form.get("smiles")

        if not smiles:
            error = "❌ Please enter a SMILES string"
            return render_template("index.html", result=result, error=error)

        if model_co2 is None:
            error = "❌ Models not loaded properly"
            return render_template("index.html", result=result, error=error)

        X = smiles_to_ecfp(smiles)

        if X is None:
            error = "❌ Invalid SMILES string"
        else:
            try:
                # ===============================
                # PREDICTIONS (REAL VALUES)
                # ===============================
                co2 = float(model_co2.predict(X)[0])
                n2  = float(model_n2.predict(X)[0])
                o2  = float(model_o2.predict(X)[0])

                # Avoid division errors
                if n2 == 0 or o2 == 0:
                    error = "❌ Division by zero in selectivity"
                    return render_template("index.html", result=result, error=error)

                # ===============================
                # SELECTIVITY
                # ===============================
                sel_co2_n2 = co2 / n2
                sel_co2_o2 = co2 / o2

                

                # ===============================
                # FINAL RESULT
                # ===============================
                result = {
                    "co2": round(co2, 6),
                    "n2": round(n2, 6),
                    "o2": round(o2, 6),

                    "sel_co2_n2": round(sel_co2_n2, 3),
                    "sel_co2_o2": round(sel_co2_o2, 3),

                   
                }

            except Exception as e:
                error = f"❌ Prediction error: {str(e)}"

    return render_template("index.html", result=result, error=error)


# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
