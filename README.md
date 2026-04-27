# Proyecto 1 - Clasificador de Tickets con Naive Bayes

Aplicacion web que clasifica tickets de mesa de ayuda usando un modelo Naive Bayes entrenado con un dataset base y ejemplos de feedback. Incluye interfaz web, API REST, y scripts para entrenamiento y evaluacion.

## Contenido del proyecto

- Backend/: API Flask, entrenamiento, y logica de NLP.
- Frontend/: interfaz web (HTML/CSS/JS) servida por Flask.
- Database/: datasets y base de datos SQLite de feedback.
- modelo_guardado/: modelo entrenado y reporte de entrenamiento.

## Requisitos

- Python 3.10 o superior (recomendado).
- Dependencias Python:
  - flask>=3.0.0
  - nltk>=3.8.1

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Nota: NLTK descarga automaticamente recursos necesarios (punkt, stopwords) al primer uso.

## Ejecucion rapida

1. Entrenar el modelo (genera modelo_guardado/modelo_nb.pkl):

```bash
python Backend/train.py
```

2. Levantar la aplicacion web:

```bash
python Backend/app.py
```

3. Abrir en el navegador:

```
http://127.0.0.1:5000
```

Opcionalmente, puedes cambiar host y puerto:

```bash
set FLASK_RUN_HOST=0.0.0.0
set FLASK_RUN_PORT=5000
python Backend/app.py
```

## Flujo de uso en la web

1. Cargar un ticket (subject y/o description).
2. Clasificar para ver categoria, probabilidad por clase y tokens.
3. Si seleccionas la categoria real, el ticket se guarda como feedback en SQLite.
4. Usar "Reentrenar IA" para recalcular el modelo con feedback acumulado.
5. (Opcional) Cargar un CSV de ejemplos para acelerar el reentrenamiento.

## Scripts disponibles

- Backend/train.py
  - Entrena el modelo con K-Fold (k=5) usando:
    - Database/bitext_dataset.csv (dataset base)
    - Database/feedback.db (feedback guardado)
    - Database/training_examples.csv (legacy, opcional)
  - Guarda el modelo y un reporte JSON con metricas.

- Backend/load_examples_and_train.py
  - Carga ejemplos desde CSV a SQLite y luego entrena.
  - Flags utiles:
    - --examples-csv <ruta>
    - --skip-load
    - --skip-train
    - --dry-run

- Backend/test_cli.py
  - Consola interactiva para probar predicciones con el modelo entrenado.

- Backend/training_summary.py
  - Muestra un resumen de clases y volumen de datos disponibles para entrenamiento.

## Formato de datos

Dataset base (Database/bitext_dataset.csv):

- instruction: texto del ticket
- category: etiqueta

CSV de ejemplos (para carga manual o upload):

- Opciones soportadas:
  - instruction, category
  - subject, description, true_category
- Columnas opcionales:
  - ticket_id, predicted_category

## API REST (Flask)

- GET /api/health
  - Estado del modelo cargado.

- POST /api/predict
  - Body JSON:
    - ticket_id (opcional)
    - subject
    - description
  - Respuesta: categoria, confianza, probabilidades, tokens.

- POST /api/feedback
  - Guarda feedback etiquetado para reentrenamiento.
  - Body JSON:
    - ticket_id, subject, description, predicted_category, true_category

- POST /api/upload-examples
  - Upload de CSV con ejemplos etiquetados.

- POST /api/train/retrain
  - Dispara reentrenamiento en segundo plano.

- GET /api/train/status
  - Estado y metricas del ultimo entrenamiento.

## Salidas generadas

- modelo_guardado/modelo_nb.pkl: modelo entrenado.
- modelo_guardado/training_report.json: metricas y matriz de confusion.
- Database/feedback.db: feedback capturado desde la interfaz o importaciones.

## Notas

- Si el modelo no existe, la UI mostrara un aviso indicando que debes entrenar.
- El procesamiento de texto aplica limpieza, stopwords y stemming (NLTK).

## Troubleshooting

- Error de modelo no encontrado:
  - Ejecuta primero `python Backend/train.py`.

- Error con recursos de NLTK:
  - Asegura acceso a internet en la primera ejecucion para descargar datos.
