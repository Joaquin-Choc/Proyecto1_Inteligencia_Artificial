from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from naive_bayes import NaiveBayesMesaAyuda
from preprocesamiento import preprocesar_ticket
from evaluacion import generar_k_folds, calcular_metricas


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "Database"
BITEXT_DATASET_PATH = DATABASE_DIR / "bitext_dataset.csv"
FEEDBACK_DB_PATH = DATABASE_DIR / "feedback.db"
LEGACY_FEEDBACK_CSV_PATH = DATABASE_DIR / "training_examples.csv"
MODEL_PATH = PROJECT_ROOT / "modelo_guardado" / "modelo_nb.pkl"


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def load_feedback_rows_from_db() -> list[dict[str, str]]:
    if not FEEDBACK_DB_PATH.exists():
        return []

    with sqlite3.connect(FEEDBACK_DB_PATH) as connection:
        cursor = connection.execute(
            """
            SELECT subject, description, true_category
            FROM feedback_samples
            """
        )
        rows = cursor.fetchall()

    feedback_rows: list[dict[str, str]] = []
    for subject, description, true_category in rows:
        feedback_rows.append(
            {
                "subject": subject or "",
                "description": description or "",
                "true_category": true_category or "",
            }
        )
    return feedback_rows


def normalize_feedback_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized_rows: list[dict[str, str]] = []

    for raw_row in rows:
        subject = (raw_row.get("subject") or "").strip()
        description = (raw_row.get("description") or "").strip()
        instruction = (raw_row.get("instruction") or "").strip()
        category = (raw_row.get("category") or raw_row.get("true_category") or "").strip().upper()

        if not description and instruction:
            description = instruction

        if not category or not (subject or description):
            continue

        text = f"{subject} {description}".strip()
        if not text:
            continue

        normalized_rows.append(
            {
                "instruction": text,
                "category": category,
            }
        )

    return normalized_rows


def load_dataset_rows() -> list[dict[str, str]]:
    rows = read_csv_rows(BITEXT_DATASET_PATH)
    if not rows:
        raise FileNotFoundError(f"No se encontró el dataset en {BITEXT_DATASET_PATH}")

    combined_rows = [
        {
            "instruction": (row.get("instruction") or "").strip(),
            "category": (row.get("category") or "").strip().upper(),
        }
        for row in rows
        if (row.get("instruction") or "").strip() and (row.get("category") or "").strip()
    ]

    combined_rows.extend(normalize_feedback_rows(load_feedback_rows_from_db()))
    combined_rows.extend(normalize_feedback_rows(read_csv_rows(LEGACY_FEEDBACK_CSV_PATH)))

    return combined_rows


def build_tokenized_dataset(rows: list[dict[str, str]]):
    tokenized_texts = []
    labels = []

    for row in rows:
        tokenized_texts.append(preprocesar_ticket(str(row["instruction"]), idioma="english"))
        labels.append(row["category"])

    return tokenized_texts, labels


def train_model(progress_callback=None):
    print("=== Sistema de Clasificación Naïve Bayes ===")
    print("1. Cargando datasets...")

    if progress_callback:
        progress_callback(5, "Cargando datasets...")

    rows = load_dataset_rows()
    if not rows:
        print("Error: no se encontraron datos para entrenamiento.")
        if progress_callback:
            progress_callback(100, "No se encontraron datos para entrenamiento.")
        return None

    print("2. Preprocesando textos (esto tomará un momento por el tamaño del dataset)...")
    if progress_callback:
        progress_callback(15, "Preprocesando textos...")
    X_completo, y_completo = build_tokenized_dataset(rows)
    clases_unicas = sorted(set(y_completo))

    print(f"Total de tickets procesados: {len(X_completo)}")
    print(f"Categorías detectadas ({len(clases_unicas)}): {clases_unicas}")

    print("\n3. Iniciando K-Folds Cross Validation (K=5)...")
    if progress_callback:
        progress_callback(25, "Iniciando validación cruzada...")
    pliegues = generar_k_folds(X_completo, y_completo, k=5)

    lista_accuracy = []
    lista_macro_f1 = []

    for i in range(5):
        print(f"   -> Evaluando Fold {i+1}/5...")
        if progress_callback:
            progress_callback(25 + (i * 12), f"Evaluando fold {i+1}/5...")
        test_data = pliegues[i]
        train_data = []
        for j in range(5):
            if i != j:
                train_data.extend(pliegues[j])

        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)

        modelo_fold = NaiveBayesMesaAyuda()
        modelo_fold.entrenar(X_train, y_train)
        y_pred = modelo_fold.predecir(X_test)

        metricas = calcular_metricas(y_test, y_pred, clases_unicas)
        lista_accuracy.append(metricas["Accuracy"])
        lista_macro_f1.append(metricas["Macro F1"])

    accuracy_promedio = sum(lista_accuracy) / len(lista_accuracy)
    macro_f1_promedio = sum(lista_macro_f1) / len(lista_macro_f1)

    print("\n=== Resultados de Evaluación (Promedio 5 Folds) ===")
    print(f"Accuracy Global: {accuracy_promedio:.4f}")
    print(f"Macro F1-Score:  {macro_f1_promedio:.4f}")
    if progress_callback:
        progress_callback(85, "Calculando métricas finales...")

    print("\n4. Entrenando modelo final con el 100% de los datos...")
    if progress_callback:
        progress_callback(92, "Entrenando modelo final...")
    modelo_final = NaiveBayesMesaAyuda()
    modelo_final.entrenar(X_completo, y_completo)
    modelo_final.guardar_modelo(MODEL_PATH)
    print(f"Modelo guardado exitosamente en: {MODEL_PATH}")
    print("¡Backend completado! Listo para integrar con la interfaz web.")
    if progress_callback:
        progress_callback(100, "Entrenamiento finalizado.")

    return {
        "total_instances": len(X_completo),
        "classes": clases_unicas,
        "accuracy": accuracy_promedio,
        "macro_f1": macro_f1_promedio,
        "model_path": str(MODEL_PATH),
    }


def main():
    return train_model()


if __name__ == '__main__':
    main()
