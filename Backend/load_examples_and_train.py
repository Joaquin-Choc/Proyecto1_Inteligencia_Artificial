from __future__ import annotations

import argparse
import csv
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "Database"
DEFAULT_EXAMPLES_CSV = DATABASE_DIR / "training_examples.csv"
FEEDBACK_DB_PATH = DATABASE_DIR / "feedback.db"


@dataclass
class ExampleRow:
    ticket_id: str
    subject: str
    description: str
    predicted_category: str
    true_category: str


def init_feedback_db() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(FEEDBACK_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                description TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                true_category TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_feedback_created_at
            ON feedback_samples(created_at)
            """
        )


def _coalesce(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def parse_example_row(raw_row: dict[str, str], row_number: int) -> ExampleRow | None:
    instruction = _coalesce(raw_row.get("instruction", ""))
    category = _coalesce(raw_row.get("category", ""), raw_row.get("true_category", "")).upper()

    subject = _coalesce(raw_row.get("subject", ""))
    description = _coalesce(raw_row.get("description", ""))

    if instruction and not description:
        description = instruction

    if not (subject or description):
        print(f"[WARN] Fila {row_number}: sin texto, se omite.")
        return None

    if not category:
        print(f"[WARN] Fila {row_number}: sin categoria, se omite.")
        return None

    ticket_id = _coalesce(raw_row.get("ticket_id", "")) or f"LOAD-{uuid.uuid4().hex[:10].upper()}"
    predicted_category = _coalesce(raw_row.get("predicted_category", "")).upper() or "MANUAL_LOAD"

    return ExampleRow(
        ticket_id=ticket_id,
        subject=subject,
        description=description,
        predicted_category=predicted_category,
        true_category=category,
    )


def load_examples_from_csv(csv_path: Path) -> list[ExampleRow]:
    if not csv_path.exists():
        print(f"[INFO] No existe archivo de ejemplos: {csv_path}")
        return []

    examples: list[ExampleRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for index, row in enumerate(reader, start=2):
            parsed = parse_example_row(row, index)
            if parsed:
                examples.append(parsed)

    return examples


def insert_examples_into_feedback_db(examples: list[ExampleRow]) -> int:
    if not examples:
        return 0

    created_at = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            example.ticket_id,
            example.subject,
            example.description,
            example.predicted_category,
            example.true_category,
            created_at,
        )
        for example in examples
    ]

    with sqlite3.connect(FEEDBACK_DB_PATH) as connection:
        connection.executemany(
            """
            INSERT INTO feedback_samples
            (ticket_id, subject, description, predicted_category, true_category, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Carga ejemplos etiquetados en feedback.db y ejecuta entrenamiento con "
            "dataset base + feedback acumulado."
        )
    )
    parser.add_argument(
        "--examples-csv",
        default=str(DEFAULT_EXAMPLES_CSV),
        help="Ruta al CSV con ejemplos (por defecto: Database/training_examples.csv)",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="No cargar ejemplos, solo entrenar con lo ya guardado en DB.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Solo cargar ejemplos, sin lanzar entrenamiento.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra lo que haria sin escribir en DB ni entrenar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.examples_csv)

    init_feedback_db()

    examples: list[ExampleRow] = []
    if not args.skip_load:
        examples = load_examples_from_csv(csv_path)
        print(f"[INFO] Ejemplos detectados para cargar: {len(examples)}")

    if args.dry_run:
        print("[DRY-RUN] No se insertan ejemplos ni se ejecuta entrenamiento.")
        return

    inserted = 0
    if not args.skip_load:
        inserted = insert_examples_into_feedback_db(examples)
        print(f"[OK] Ejemplos insertados en feedback.db: {inserted}")

    if args.skip_train:
        print("[INFO] Entrenamiento omitido por bandera --skip-train.")
        return

    print("[INFO] Iniciando entrenamiento...")
    try:
        import train
    except ModuleNotFoundError as exc:
        print(
            "[ERROR] No se pudo iniciar entrenamiento por dependencias faltantes. "
            f"Detalle: {exc}.\n"
            "Instala dependencias del entrenamiento y vuelve a ejecutar."
        )
        return

    train.main()
    print("[OK] Entrenamiento finalizado.")


if __name__ == "__main__":
    main()
