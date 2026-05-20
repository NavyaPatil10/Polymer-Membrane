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
    model_n2  = joblib.load("final_xgb_N2.pkl")
    model_o2  = joblib.load("final_xgb_O2.pkl")

    print("✅ Models loaded successfully")

except Exception as e:

    print("❌ Model loading error:", e)

    model_co2 = None
    model_n2 = None
    model_o2 = None


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

    # Fast conversion
    arr = np.array(fp).reshape(1, -1)

    return arr


# =========================================
# ROUTE
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

        # =========================================
        # EMPTY INPUT
        # =========================================
        if not smiles:

            error = "❌ Please enter a SMILES string"

            return render_template(
                "index.html",
                result=result,
                error=error
            )

        # =========================================
        # MODEL CHECK
        # =========================================
        if model_co2 is None:

            error = "❌ Models not loaded properly"

            return render_template(
                "index.html",
                result=result,
                error=error
            )

        # =========================================
        # CONVERT TO ECFP
        # =========================================
        X = smiles_to_ecfp(smiles)

        # =========================================
        # INVALID SMILES
        # =========================================
        if X is None:

            error = "❌ Invalid SMILES string"

        else:

            try:

                # =========================================
                # PREDICTIONS
                # =========================================
                co2 = float(model_co2.predict(X)[0])

                n2 = float(model_n2.predict(X)[0])

                o2 = float(model_o2.predict(X)[0])

                # =========================================
                # AVOID DIVISION ERROR
                # =========================================
                if n2 == 0 or o2 == 0:

                    error = "❌ Division by zero in selectivity"

                    return render_template(
                        "index.html",
                        result=result,
                        error=error
                    )

                # =========================================
                # SELECTIVITY
                # =========================================
                sel_co2_n2 = co2 / n2

                sel_co2_o2 = co2 / o2

                # =========================================
                # INTERPRETATION + APPLICATION
                # =========================================

                # Strong CO2/N2 and strong CO2/O2
                if sel_co2_n2 >= 30 and sel_co2_o2 >= 10:

                    interpretation = """
                    The membrane demonstrates excellent carbon dioxide
                    separation capability against both nitrogen and oxygen
                    gases, indicating strong potential for advanced carbon
                    capture and industrial gas separation applications.
                    """

                    application = """
                    Carbon Capture • Industrial Gas Separation •
                    Environmental Protection Systems
                    """

                # Strong CO2/N2 but moderate CO2/O2
                elif sel_co2_n2 >= 30 and sel_co2_o2 < 10:

                    interpretation = """
                    The membrane demonstrates strong carbon dioxide
                    separation from nitrogen gases, while oxygen
                    separation performance is comparatively moderate.
                    This makes the membrane suitable for flue gas
                    treatment and carbon capture applications.
                    """

                    application = """
                    Flue Gas Separation • Carbon Capture •
                    Industrial CO₂ Recovery
                    """

                # Moderate performance
                elif sel_co2_n2 >= 10 or sel_co2_o2 >= 5:

                    interpretation = """
                    The membrane demonstrates moderate gas separation
                    characteristics and may be suitable for selective
                    gas purification applications.
                    """

                    application = """
                    Gas Purification • Membrane Filtration •
                    Industrial Separation Systems
                    """

                # Low performance
                else:

                    interpretation = """
                    The membrane shows lower gas separation efficiency
                    and may require further optimization for practical
                    industrial deployment.
                    """

                    application = """
                    Research and Material Optimization
                    """

                # =========================================
                # FINAL RESULT
                # =========================================
                result = {

                    "co2": round(co2, 6),

                    "n2": round(n2, 6),

                    "o2": round(o2, 6),

                    "sel_co2_n2": round(sel_co2_n2, 3),

                    "sel_co2_o2": round(sel_co2_o2, 3)
                }

            except Exception as e:

                error = f"❌ Prediction error: {str(e)}"

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
