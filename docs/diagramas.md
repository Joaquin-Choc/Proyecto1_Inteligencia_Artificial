# Diagramas del Sistema — Clasificación de Solicitudes a Mesa de Ayuda

## 1. Arquitectura de la Solución

```mermaid
graph TB
    subgraph FRONTEND["Frontend (HTML + CSS + JS)"]
        UI["index.html\nPortal de Soporte"]
        JS["app.js\nLógica de UI"]
        CHART["Chart.js\nVisualización"]
        UI --> JS
        JS --> CHART
    end

    subgraph BACKEND["Backend (Flask - app.py)"]
        FLASK["Servidor Flask"]
        subgraph ROUTES["API Endpoints"]
            R1["GET /\nPágina principal"]
            R2["POST /api/predict\nClasificar ticket"]
            R3["POST /api/feedback\nGuardar feedback"]
            R4["GET /api/train/status\nEstado entrenamiento"]
            R5["POST /api/train/retrain\nReiniciar entrenamiento"]
            R6["POST /api/upload-examples\nSubir CSV"]
        end
        FLASK --> ROUTES
    end

    subgraph ENGINE["Motor de Inferencia"]
        PREP["preprocesamiento.py\nlimpiar_texto()\npreprocesar_ticket()"]
        NB["naive_bayes.py\nNaiveBayesMesaAyuda\nentrenar() / predecir_proba()"]
        TRAIN["train.py\ntrain_model()\nK-Folds CV"]
        EVAL["evaluacion.py\ncalcular_metricas()\nconfusion_matrix"]
        PREP --> NB
        TRAIN --> PREP
        TRAIN --> NB
        TRAIN --> EVAL
    end

    subgraph STORAGE["Almacenamiento"]
        PKL["modelo_nb.pkl\nModelo serializado"]
        JSON["training_report.json\nMétricas K-Folds"]
        CSV["bitext_dataset.csv\n53,751 registros"]
        DB[("SQLite\nfeedback_samples")]
    end

    JS -- "HTTP REST" --> FLASK
    R2 --> PREP
    R2 --> NB
    R5 --> TRAIN
    R3 --> DB
    NB -- "guardar_modelo()" --> PKL
    TRAIN -- "cargar datos" --> CSV
    TRAIN -- "cargar feedback" --> DB
    TRAIN -- "genera reporte" --> JSON
    NB -- "cargar_modelo()" --> PKL
    FLASK -- "carga al inicio" --> PKL
```

---

## 2. Diagrama de Casos de Uso

```mermaid
graph LR
    USER(["Usuario / Cliente"])
    ADMIN(["Administrador / Analista"])

    subgraph SISTEMA["Sistema de Clasificación de Tickets"]
        UC1["Ingresar solicitud\n(subject + description)"]
        UC2["Obtener categoría predicha"]
        UC3["Ver probabilidades\npor clase"]
        UC4["Ver tokens\nprocesados"]
        UC5["Proporcionar categoría\ncorrecta (feedback)"]
        UC6["Generar ID de Ticket\nautomático"]

        UC7["Subir ejemplos\nen CSV"]
        UC8["Iniciar reentrenamiento\ndel modelo"]
        UC9["Monitorear progreso\ndel entrenamiento"]
        UC10["Ver métricas\nK-Folds"]
        UC11["Ver matriz de\nconfusión"]

        UC2 -.->|"include"| UC1
        UC3 -.->|"include"| UC2
        UC4 -.->|"include"| UC2
        UC5 -.->|"extend"| UC2
        UC6 -.->|"include"| UC1

        UC8 -.->|"include"| UC9
        UC10 -.->|"include"| UC8
        UC7 -.->|"extend"| UC8
    end

    USER --> UC1
    USER --> UC5
    USER --> UC3

    ADMIN --> UC7
    ADMIN --> UC8
    ADMIN --> UC9
    ADMIN --> UC10
    ADMIN --> UC11
```

---

## 3. Diagrama de Flujo General

```mermaid
flowchart TD
    START([Inicio]) --> LOAD["Cargar modelo\nmodelo_nb.pkl"]
    LOAD --> MODEL_OK{¿Modelo\ncargado?}
    MODEL_OK -- No --> ERROR["Mostrar error\nen UI"]
    MODEL_OK -- Sí --> READY["Sistema listo\nGET /"]

    READY --> INPUT["Usuario ingresa ticket\nticket_id, subject, description"]
    INPUT --> VALIDATE{¿Texto\nválido?}
    VALIDATE -- No --> WARN["Mostrar advertencia\nPedir más texto"]
    WARN --> INPUT
    VALIDATE -- Sí --> PREPROCESS

    subgraph PREPROCESS["Preprocesamiento"]
        P1["limpiar_texto()\nRemover {{...}}, especiales, lowercase"]
        P2["word_tokenize()\nNLTK tokenización"]
        P3["Eliminar stopwords\nEnglish stopwords"]
        P4["SnowballStemmer()\nStemming"]
        P1 --> P2 --> P3 --> P4
    end

    PREPROCESS --> INFER

    subgraph INFER["Inferencia Naïve Bayes"]
        I1["calcular_scores()\nlog P(clase) + Σ log P(token|clase)"]
        I2["Laplace Smoothing\nP = (freq+1) / (total+vocab_size)"]
        I3["predecir_proba()\nSoftmax de log-scores"]
        I1 --> I2 --> I3
    end

    INFER --> RESULT["Retornar predicción\nclase + confianza + probabilidades"]
    RESULT --> SHOW["Mostrar resultado\nChart.js + barras"]

    SHOW --> FEEDBACK{¿Usuario da\nfeedback?}
    FEEDBACK -- No --> END_CLASSIFY([Fin clasificación])
    FEEDBACK -- Sí --> SAVE_FB["POST /api/feedback\nGuardar en SQLite"]
    SAVE_FB --> RETRAIN{¿Reentrenar\nahora?}
    RETRAIN -- No --> END_CLASSIFY
    RETRAIN -- Sí --> TRAIN_FLOW

    subgraph TRAIN_FLOW["Entrenamiento K-Folds"]
        T1["Cargar dataset\nCSV + SQLite feedback"]
        T2["Tokenizar\ntodo el corpus"]
        T3["Dividir en 5 folds\nK=5, seed=42"]
        T4["Para cada fold:\nEntrenar 80% → Evaluar 20%"]
        T5["Acumular métricas\nPrecision, Recall, F1"]
        T6["Entrenar modelo final\n100% datos"]
        T7["Guardar modelo_nb.pkl\n+ training_report.json"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7
    end

    RETRAIN -- Sí --> T1
    T7 --> READY
```

---

## 4. Diagrama de Componentes

```mermaid
graph TB
    subgraph CLIENT["Capa de Presentación"]
        direction TB
        HTML["&lt;&lt;component&gt;&gt;\nindex.html\nPlantilla Jinja2"]
        JS_MOD["&lt;&lt;component&gt;&gt;\napp.js\nLógica de interfaz"]
        CSS["&lt;&lt;component&gt;&gt;\nstyles.css\nEstilos"]
        CHARTJS["&lt;&lt;library&gt;&gt;\nChart.js\nGraficación"]
        BOOTSTRAP["&lt;&lt;library&gt;&gt;\nBootstrap 5\nUI Framework"]

        HTML --> JS_MOD
        HTML --> CSS
        JS_MOD --> CHARTJS
        HTML --> BOOTSTRAP
    end

    subgraph SERVER["Capa de Aplicación"]
        direction TB
        APP["&lt;&lt;component&gt;&gt;\napp.py\nFlask Application\n(Routes + State)"]
        LOAD_CLI["&lt;&lt;component&gt;&gt;\nload_examples_and_train.py\nCLI Utility"]
    end

    subgraph INFERENCE["Capa de Inferencia"]
        direction TB
        PREPROC["&lt;&lt;component&gt;&gt;\npreprocesamiento.py\nlimpiar_texto()\npreprocesar_ticket()"]
        NAIVE["&lt;&lt;component&gt;&gt;\nnaive_bayes.py\nNaiveBayesMesaAyuda\nBag of Words + Laplace"]
        TRAINER["&lt;&lt;component&gt;&gt;\ntrain.py\nPipeline de entrenamiento\nK-Folds CV"]
        EVALUATOR["&lt;&lt;component&gt;&gt;\nevaluacion.py\nMétricas + Matriz\nde Confusión"]
    end

    subgraph PERSISTENCE["Capa de Persistencia"]
        direction LR
        PICKLE["&lt;&lt;file&gt;&gt;\nmodelo_nb.pkl\nModelo serializado"]
        REPORT["&lt;&lt;file&gt;&gt;\ntraining_report.json\nMétricas K-Folds"]
        DATASET["&lt;&lt;file&gt;&gt;\nbitext_dataset.csv\n53,751 instancias"]
        SQLITE[("&lt;&lt;database&gt;&gt;\nSQLite\nfeedback_samples")]
    end

    subgraph NLTK_LIB["Librerías NLP (NLTK)"]
        TOK["word_tokenize"]
        STOP["stopwords (English)"]
        STEM["SnowballStemmer"]
    end

    JS_MOD -- "HTTP/REST\nJSON" --> APP
    APP --> PREPROC
    APP --> NAIVE
    APP --> TRAINER
    APP --> SQLITE

    PREPROC --> TOK
    PREPROC --> STOP
    PREPROC --> STEM

    TRAINER --> PREPROC
    TRAINER --> NAIVE
    TRAINER --> EVALUATOR
    TRAINER --> DATASET
    TRAINER --> SQLITE

    NAIVE -- "guardar_modelo()" --> PICKLE
    NAIVE -- "cargar_modelo()" --> PICKLE
    TRAINER --> REPORT
    APP -- "load_model()" --> PICKLE
    APP -- "load_training_state()" --> REPORT
    LOAD_CLI --> SQLITE
    LOAD_CLI --> TRAINER
```
