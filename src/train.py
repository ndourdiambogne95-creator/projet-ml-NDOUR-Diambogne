"""
Projet 2 — Détection d'attaques réseau par Machine Learning
Entraînement, comparaison des modèles et sauvegarde du meilleur.

Lancement (depuis la racine du dépôt) :  python src/train.py
"""

import json
import unicodedata
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "flux_train.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "model.joblib"
FIG_DIR = BASE_DIR / "figures"

CIBLE = "attaque"
COLS_NUM = [
    "duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
    "nb_failed_logins", "nb_conn_same_host", "syn_rate",
]
COLS_CAT = ["protocol"]
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Exploration
# ---------------------------------------------------------------------------
def explorer(df: pd.DataFrame) -> None:
    print("=== Exploration ===")
    print("Taille :", df.shape)
    print("Valeurs manquantes :", df.isna().sum()[df.isna().sum() > 0].to_dict())
    print("Répartition de la cible :")
    print(df[CIBLE].value_counts().rename({0: "normal", 1: "attaque"}).to_string())
    print(f"=> {df[CIBLE].mean():.1%} d'attaques (classes déséquilibrées)\n")


# ---------------------------------------------------------------------------
# 2. Prétraitement + modèles (tout dans un Pipeline)
# ---------------------------------------------------------------------------
def construire_pretraitement() -> ColumnTransformer:
    numerique = Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("echelle", StandardScaler()),
    ])
    categoriel = Pipeline([
        ("imputation", SimpleImputer(strategy="most_frequent")),
        ("encodage", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numerique, COLS_NUM),
        ("cat", categoriel, COLS_CAT),
    ])


def construire_modeles() -> dict:
    classifieurs = {
        "Régression logistique": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Arbre de décision": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
        "Forêt aléatoire": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    return {
        nom: Pipeline([("pretraitement", construire_pretraitement()), ("modele", clf)])
        for nom, clf in classifieurs.items()
    }


# ---------------------------------------------------------------------------
# 3. Évaluation
# ---------------------------------------------------------------------------
def evaluer(nom, pipe, X_train, X_val, y_train, y_val, X, y) -> dict:
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f1_cv = cross_val_score(pipe, X, y, cv=cv, scoring="f1")

    cm = confusion_matrix(y_val, y_pred)
    print(f"=== {nom} ===")
    print(classification_report(y_val, y_pred, target_names=["normal", "attaque"], digits=3))
    print("Matrice de confusion [[VN, FP], [FN, VP]] :")
    print(cm)
    print(f"F1 validation croisée (5 plis) : {f1_cv.mean():.3f} ± {f1_cv.std():.3f}\n")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["normal", "attaque"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matrice de confusion — {nom}")
    fig.tight_layout()
    slug = unicodedata.normalize("NFKD", nom).encode("ascii", "ignore").decode()
    fichier = "confusion_" + slug.lower().replace(" ", "_")
    fig.savefig(FIG_DIR / f"{fichier}.png", dpi=120)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    return {
        "f1": round(f1_score(y_val, y_pred), 4),
        "rappel": round(recall_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred), 4),
        "f1_cv_moyen": round(float(f1_cv.mean()), 4),
        "f1_cv_ecart_type": round(float(f1_cv.std()), 4),
        "vrais_negatifs": int(tn), "faux_positifs": int(fp),
        "faux_negatifs": int(fn), "vrais_positifs": int(tp),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    explorer(df)

    X = df.drop(columns=[CIBLE])
    y = df[CIBLE]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    resultats = {}
    for nom, pipe in construire_modeles().items():
        resultats[nom] = evaluer(nom, pipe, X_train, X_val, y_train, y_val, X, y)

    tableau = pd.DataFrame(resultats).T
    print("=== Comparaison ===")
    print(tableau[["f1", "rappel", "precision", "f1_cv_moyen", "faux_negatifs", "faux_positifs"]])

    # Choix : meilleur F1 moyen en validation croisée (plus fiable qu'un seul découpage)
    meilleur = tableau["f1_cv_moyen"].astype(float).idxmax()
    print(f"\nModèle retenu : {meilleur}")

    # Ré-entraînement sur toutes les données puis sauvegarde
    final = construire_modeles()[meilleur]
    final.fit(X, y)
    joblib.dump(final, MODEL_FILE)
    print(f"Modèle sauvegardé dans {MODEL_FILE}")

    with open(MODEL_DIR / "metriques.json", "w", encoding="utf-8") as f:
        json.dump({"modele_retenu": meilleur, "resultats": resultats}, f,
                  indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
