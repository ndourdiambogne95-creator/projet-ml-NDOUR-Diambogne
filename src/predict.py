"""
Projet 2 — Prédiction avec le modèle entraîné.

Utilisation :
    from src.predict import predict
    predictions = predict(df)          # df : colonnes de flux_train.csv sans 'attaque'

ou en ligne de commande :
    python src/predict.py data/flux_train.csv
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "model.joblib"

COLONNES = [
    "duration", "protocol", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
    "nb_failed_logins", "nb_conn_same_host", "syn_rate",
]

_modele = None  # chargé une seule fois


def _charger_modele():
    """Charge models/model.joblib (Pipeline prétraitement + modèle).

    Sécurité : si le fichier est absent ou incompatible avec la version de
    scikit-learn installée, le modèle est ré-entraîné à partir de src/train.py.
    """
    global _modele
    if _modele is None:
        try:
            _modele = joblib.load(MODEL_FILE)
        except Exception as erreur:  # fichier manquant ou version incompatible
            print(f"[predict] Chargement impossible ({erreur}) -> ré-entraînement.")
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import train
            train.main()
            _modele = joblib.load(MODEL_FILE)
    return _modele


def _preparer(df: pd.DataFrame) -> pd.DataFrame:
    """Garde les colonnes attendues, dans le bon ordre, et harmonise le protocole."""
    X = df.drop(columns=["attaque"], errors="ignore")[COLONNES].copy()
    X["protocol"] = X["protocol"].astype(str).str.strip().str.upper()
    return X


def predict(df: pd.DataFrame) -> list:
    """Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne de df."""
    modele = _charger_modele()
    predictions = modele.predict(_preparer(df))
    return [int(p) for p in predictions]


if __name__ == "__main__":
    fichier = sys.argv[1] if len(sys.argv) > 1 else BASE_DIR / "data" / "flux_train.csv"
    donnees = pd.read_csv(fichier)
    preds = predict(donnees)
    print(f"{len(preds)} flux analysés : {sum(preds)} attaques détectées")
    if "attaque" in donnees.columns:
        from sklearn.metrics import f1_score
        print(f"F1 (attaque) sur ce fichier : {f1_score(donnees['attaque'], preds):.3f}")
