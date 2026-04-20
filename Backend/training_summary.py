from __future__ import annotations

import csv
import sqlite3
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "Database"
DATASET_PATH = DATABASE_DIR / "bitext_dataset.csv"
FEEDBACK_DB_PATH = DATABASE_DIR / "feedback.db"
LEGACY_FEEDBACK_CSV_PATH = DATABASE_DIR / "training_examples.csv"


def read_base_dataset_summary() -> tuple[int, Counter]:
    class_counter: Counter[str] = Counter()
    total_rows = 0

    with DATASET_PATH.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            category = (row.get("category") or "").strip().upper()
            if not category:
                continue
            total_rows += 1
            class_counter[category] += 1

    return total_rows, class_counter


def read_feedback_db_summary() -> tuple[int, Counter]:
    if not FEEDBACK_DB_PATH.exists():
        return 0, Counter()

    with sqlite3.connect(FEEDBACK_DB_PATH) as connection:
        rows = connection.execute(
            "SELECT true_category FROM feedback_samples"
        ).fetchall()

    class_counter: Counter[str] = Counter()
    for (true_category,) in rows:
        value = (true_category or "").strip().upper()
        if value:
            class_counter[value] += 1

    return len(rows), class_counter


def read_legacy_feedback_summary() -> tuple[int, Counter]:
    if not LEGACY_FEEDBACK_CSV_PATH.exists():
        return 0, Counter()

    class_counter: Counter[str] = Counter()
    total_rows = 0

    with LEGACY_FEEDBACK_CSV_PATH.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            category = (row.get("true_category") or "").strip().upper()
            if not category:
                continue
            total_rows += 1
            class_counter[category] += 1

    return total_rows, class_counter


def main() -> None:
    base_total, base_classes = read_base_dataset_summary()
    feedback_total, feedback_classes = read_feedback_db_summary()
    legacy_total, legacy_classes = read_legacy_feedback_summary()

    merged_classes = base_classes + feedback_classes + legacy_classes
    merged_total = base_total + feedback_total + legacy_total

    print("=== Resumen de entrenamiento ===")
    print(f"Dataset base: {base_total} instancias")
    print(f"Feedback SQLite: {feedback_total} instancias")
    print(f"Feedback CSV legado: {legacy_total} instancias")
    print(f"Total potencial para entrenamiento: {merged_total} instancias")
    print("")
    print("Top clases (dataset combinado):")

    for category, count in merged_classes.most_common(12):
        print(f"- {category}: {count}")


if __name__ == "__main__":
    main()
