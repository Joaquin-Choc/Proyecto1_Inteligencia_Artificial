from __future__ import annotations

import csv
import io
import json
import os
import random
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from naive_bayes import NaiveBayesMesaAyuda
from preprocesamiento import preprocesar_ticket


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "Frontend"
MODEL_PATH = PROJECT_ROOT / "modelo_guardado" / "modelo_nb.pkl"
REPORT_PATH = PROJECT_ROOT / "modelo_guardado" / "training_report.json"
FEEDBACK_DB_PATH = PROJECT_ROOT / "Database" / "feedback.db"

app = Flask(
	__name__,
	template_folder=str(FRONTEND_DIR / "templates"),
	static_folder=str(FRONTEND_DIR / "static"),
)

model = NaiveBayesMesaAyuda()
model_loaded = False
load_error = None
training_state = {
	"status": "idle",
	"message": "",
	"progress": 0,
	"result": None,
	"error": None,
}
training_lock = threading.Lock()


def init_feedback_db() -> None:
	FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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


def load_model() -> None:
	global model_loaded, load_error

	if not MODEL_PATH.exists():
		model_loaded = False
		load_error = (
			f"No se encontró el modelo en {MODEL_PATH}. Ejecuta primero Backend/train.py."
		)
		return

	try:
		model.cargar_modelo(str(MODEL_PATH))
		model_loaded = True
		load_error = None
	except Exception as exc:  # pragma: no cover - fallback para error de entorno
		model_loaded = False
		load_error = f"No se pudo cargar el modelo: {exc}"


def generate_ticket_id() -> str:
	return f"TKT-{random.randint(100000, 999999)}"


def save_feedback_sample(
	ticket_id: str,
	subject: str,
	description: str,
	predicted_category: str,
	true_category: str,
) -> int:
	created_at = datetime.now(timezone.utc).isoformat()
	with sqlite3.connect(FEEDBACK_DB_PATH) as connection:
		cursor = connection.execute(
			"""
			INSERT INTO feedback_samples
			(ticket_id, subject, description, predicted_category, true_category, created_at)
			VALUES (?, ?, ?, ?, ?, ?)
			""",
			(ticket_id, subject, description, predicted_category, true_category, created_at),
		)
		connection.commit()
		return int(cursor.lastrowid)


def classify_ticket(subject: str, description: str):
	combined_text = " ".join(part.strip() for part in [subject, description] if part and part.strip())
	tokens = preprocesar_ticket(combined_text, idioma="english")

	if not tokens:
		return {
			"prediction": None,
			"confidence": 0.0,
			"probabilities": {},
			"tokens": [],
			"combined_text": combined_text,
		}

	probabilities = model.predecir_proba(tokens)
	prediction = max(probabilities, key=probabilities.get) if probabilities else model.predecir_instancia(tokens)
	confidence = probabilities.get(prediction, 0.0)

	return {
		"prediction": prediction,
		"confidence": confidence,
		"probabilities": probabilities,
		"tokens": tokens,
		"combined_text": combined_text,
	}


def is_valid_true_category(true_category: str) -> bool:
	return bool(true_category and true_category in model.clases)


def is_valid_category(category: str) -> bool:
	return bool(category and category in model.clases)


def set_training_state(status: str, message: str = "", progress: int | None = None, result=None, error=None) -> None:
	with training_lock:
		training_state["status"] = status
		training_state["message"] = message
		if progress is not None:
			training_state["progress"] = max(0, min(100, int(progress)))
		training_state["result"] = result
		training_state["error"] = error


def load_training_state_from_report() -> None:
	if not REPORT_PATH.exists():
		return

	try:
		with REPORT_PATH.open("r", encoding="utf-8") as report_file:
			report_data = json.load(report_file)
		set_training_state(
			"completed",
			"Reporte de entrenamiento cargado.",
			progress=100,
			result=report_data,
		)
	except Exception:
		# Si el reporte no puede leerse, mantenemos el estado por defecto sin bloquear la app.
		return


def run_training_job() -> None:
	try:
		set_training_state("running", "Entrenando el modelo...", progress=1)
		import train

		def progress_callback(progress: int, message: str) -> None:
			set_training_state("running", message, progress=progress)

		result = train.train_model(progress_callback=progress_callback)
		load_model()
		set_training_state("completed", "Entrenamiento finalizado.", progress=100, result=result)
	except Exception as exc:  # pragma: no cover - se reporta al frontend
		set_training_state("failed", "El entrenamiento falló.", progress=100, error=str(exc))


def parse_uploaded_example_row(raw_row: dict[str, str], row_number: int):
	instruction = (raw_row.get("instruction") or "").strip()
	category = (raw_row.get("category") or raw_row.get("true_category") or "").strip().upper()
	subject = (raw_row.get("subject") or "").strip()
	description = (raw_row.get("description") or "").strip()
	predicted_category = (raw_row.get("predicted_category") or "MANUAL_LOAD").strip().upper()

	if not description and instruction:
		description = instruction

	if not subject and not description:
		print(f"[WARN] Fila {row_number}: sin texto, se omite.")
		return None

	if not category:
		print(f"[WARN] Fila {row_number}: sin categoria, se omite.")
		return None

	if not is_valid_true_category(category):
		print(f"[WARN] Fila {row_number}: categoria no valida ({category}), se omite.")
		return None

	return {
		"ticket_id": (raw_row.get("ticket_id") or generate_ticket_id()).strip(),
		"subject": subject,
		"description": description,
		"predicted_category": predicted_category,
		"true_category": category,
	}


@app.route("/")
def index():
	return render_template(
		"index.html",
		model_loaded=model_loaded,
		load_error=load_error,
		classes=sorted(model.clases) if model_loaded else [],
		ticket_id=generate_ticket_id(),
	)


@app.route("/api/health")
def health():
	return jsonify(
		{
			"status": "ok" if model_loaded else "degraded",
			"model_loaded": model_loaded,
			"error": load_error,
		}
	)


@app.route("/api/predict", methods=["POST"])
def predict():
	if not model_loaded:
		return (
			jsonify(
				{
					"error": load_error or "El modelo no está cargado.",
				}
			),
			503,
		)

	payload = request.get_json(silent=True) or {}
	ticket_id = (payload.get("ticket_id") or generate_ticket_id()).strip()
	subject = (payload.get("subject") or "").strip()
	description = (payload.get("description") or "").strip()
	comment = (payload.get("comment") or "").strip()

	if comment and not (subject or description):
		description = comment

	if not (subject or description):
		return jsonify({"error": "Debes ingresar subject o description."}), 400

	result = classify_ticket(subject, description)
	result.update(
		{
			"ticket_id": ticket_id,
			"subject": subject,
			"description": description,
			"classes": sorted(model.clases),
		}
	)

	return jsonify(result)


@app.route("/api/feedback", methods=["POST"])
def save_feedback():
	if not model_loaded:
		return (
			jsonify({"error": load_error or "El modelo no está cargado."}),
			503,
		)

	payload = request.get_json(silent=True) or {}
	ticket_id = (payload.get("ticket_id") or "").strip()
	subject = (payload.get("subject") or "").strip()
	description = (payload.get("description") or "").strip()
	predicted_category = (payload.get("predicted_category") or "").strip().upper()
	true_category = (payload.get("true_category") or "").strip().upper()

	if not all([ticket_id, subject, description, predicted_category, true_category]):
		return jsonify({"error": "Faltan campos obligatorios para guardar feedback."}), 400

	if not is_valid_category(predicted_category):
		return jsonify({"error": "La categoría predicha no es válida."}), 400

	if not is_valid_true_category(true_category):
		return jsonify({"error": "La categoría real no es válida."}), 400

	feedback_id = save_feedback_sample(
		ticket_id=ticket_id,
		subject=subject,
		description=description,
		predicted_category=predicted_category,
		true_category=true_category,
	)

	return jsonify({"feedback_id": feedback_id, "saved": True}), 201


@app.route("/api/train/status")
def train_status():
	with training_lock:
		return jsonify(training_state)


@app.route("/api/train/retrain", methods=["POST"])
def retrain_model():
	with training_lock:
		if training_state["status"] == "running":
			return jsonify({"error": "Ya hay un reentrenamiento en curso."}), 409

	set_training_state("running", "Iniciando reentrenamiento...")
	thread = threading.Thread(target=run_training_job, daemon=True)
	thread.start()
	return jsonify({"started": True, "status": "running"}), 202


@app.route("/api/upload-examples", methods=["POST"])
def upload_examples():
	if not model_loaded:
		return (
			jsonify({"error": load_error or "El modelo no está cargado."}),
			503,
		)

	file = request.files.get("file")
	if not file:
		return jsonify({"error": "Debes seleccionar un archivo CSV."}), 400

	raw_text = file.stream.read().decode("utf-8-sig", errors="replace")
	reader = csv.DictReader(io.StringIO(raw_text))

	rows_to_save = []
	for row_number, raw_row in enumerate(reader, start=2):
		parsed = parse_uploaded_example_row(raw_row, row_number)
		if parsed:
			rows_to_save.append(parsed)

	if not rows_to_save:
		return jsonify({"error": "El archivo no contenia ejemplos validos."}), 400

	created_at = datetime.now(timezone.utc).isoformat()
	with sqlite3.connect(FEEDBACK_DB_PATH) as connection:
		connection.executemany(
			"""
			INSERT INTO feedback_samples
			(ticket_id, subject, description, predicted_category, true_category, created_at)
			VALUES (?, ?, ?, ?, ?, ?)
			""",
			[
				(
					row["ticket_id"],
					row["subject"],
					row["description"],
					row["predicted_category"],
					row["true_category"],
					created_at,
				)
				for row in rows_to_save
			],
		)
		connection.commit()

	return jsonify(
		{
			"saved": True,
			"imported_count": len(rows_to_save),
			"filename": file.filename,
		}
	), 201


init_feedback_db()
load_model()
load_training_state_from_report()


if __name__ == "__main__":
	host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
	port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
	app.run(debug=True, host=host, port=port)
