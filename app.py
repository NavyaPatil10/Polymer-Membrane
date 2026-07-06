from flask import Flask, render_template, request
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)

# =========================================
# LOAD MODELS
# =========================================
try:
    model_co2 = joblib.load("final_xgb_CO2.pkl")
    model_n2 = joblib.load("final_xgb_N2.pkl")

    print("✅ Models loaded successfully")

except Exception as e:

    print("❌ Model loading error:", e)

    model_co2 = None
    model_n2 = None


# =========================================
# SMILES → ECFP
# =========================================
def smiles_to_ecfp(smiles, n_bits=2048):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=n_bits
    )

    arr = np.array(fp).reshape(1, -1)

    return arr


# =========================================
# HOME
# =========================================
@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    interpretation = None
    application = None
    smiles = None

    if request.method == "POST":

        smiles = request.form.get("smiles")

        # Empty input
        if not smiles:

            error = "Please enter a SMILES string."

            return render_template(
                "index.html",
                error=error
            )

        # Check models
        if model_co2 is None or model_n2 is None:

            error = "Models not loaded properly."

            return render_template(
                "index.html",
                error=error
            )

        # Convert to fingerprint
        X = smiles_to_ecfp(smiles)

        if X is None:

            error = "Invalid SMILES string."

        else:

            try:

                # Predictions
                c# =========================================
# LOG PREDICTIONS
# =========================================
                log_co2 = float(model_co2.predict(X)[0])
                log_n2 = float(model_n2.predict(X)[0])

# =========================================
# CONVERT TO PERMEABILITY (Barrer)
# =========================================
                co2 = 10 ** log_co2
                n2 = 10 ** log_n2

                if n2 <= 0:

                    error = "Prediction resulted in invalid N₂ permeability."

                else:

    # =========================================
    # CO₂/N₂ SELECTIVITY
    # =========================================
                    selectivity = co2 / n2

                    # Interpretation
                    if selectivity >= 30:

                        interpretation = (
                            "The polymer membrane exhibits excellent "
                            "CO₂/N₂ separation performance with high "
                            "potential for carbon capture applications."
                        )

                        application = (
                            "Carbon Capture • Flue Gas Separation • "
                            "Industrial CO₂ Recovery"
                        )

                    elif selectivity >= 15:

                        interpretation = (
                            "The membrane demonstrates good CO₂/N₂ "
                            "separation performance suitable for several "
                            "industrial gas separation processes."
                        )

                        application = (
                            "Gas Separation • Industrial Membranes • "
                            "CO₂ Enrichment"
                        )

                    elif selectivity >= 10:

                        interpretation = (
                            "The membrane exhibits moderate separation "
                            "performance and may require optimization "
                            "for practical deployment."
                        )

                        application = (
                            "Membrane Development • Material Optimization"
                        )

                    else:

                        interpretation = (
                            "The membrane exhibits relatively low "
                            "CO₂/N₂ selectivity and may require further "
                            "material design improvements."
                        )

                        application = (
                            "Research and Material Optimization"
                        )

                    result = {

                        "co2": round(co2, 6),

                        "n2": round(n2, 6),

                        "selectivity": round(selectivity, 3)

                    }

            except Exception as e:

                error = f"Prediction error: {str(e)}"

    return render_template(

        "index.html",

        result=result,

        error=error,

        interpretation=interpretation,

        application=application,

        smiles=smiles

    )


# =========================================
# RUN
# =========================================
if __name__ == "__main__":

    app.run(debug=True)
