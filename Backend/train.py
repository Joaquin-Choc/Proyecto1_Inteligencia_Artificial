import pandas as pd
from naive_bayes import NaiveBayesMesaAyuda
from preprocesamiento import preprocesar_ticket
from evaluacion import generar_k_folds, calcular_metricas

def main():
    print("=== Sistema de Clasificación Naïve Bayes ===")
    print("1. Cargando dataset de Bitext...")
    
    # 1. Ruta al nuevo dataset
    ruta_dataset = 'data/bitext_dataset.csv'
    try:
        df = pd.read_csv(ruta_dataset)
    except FileNotFoundError:
        print(f"Error: No se encontró el dataset en {ruta_dataset}. Verifica la ruta.")
        return

    # 2. Definir las columnas exactas del nuevo CSV
    columna_texto = 'instruction' 
    columna_etiqueta = 'category'

    # Eliminamos filas nulas por si acaso
    df = df.dropna(subset=[columna_texto, columna_etiqueta])

    print("2. Preprocesando textos (esto tomará un momento por el tamaño del dataset)...")
    
    # Aplicamos la limpieza, eliminación de placeholders, tokenización y stemming
    df['Tokens'] = df[columna_texto].apply(lambda x: preprocesar_ticket(str(x), idioma='english'))
    
    # Extraemos las listas para inyectarlas al modelo
    X_completo = df['Tokens'].tolist()
    y_completo = df[columna_etiqueta].tolist()
    clases_unicas = list(set(y_completo))

    print(f"Total de tickets procesados: {len(X_completo)}")
    print(f"Categorías detectadas ({len(clases_unicas)}): {clases_unicas}")

    # ==========================================
    # FASE DE EVALUACIÓN: K-Folds (K=5)
    # ==========================================
    print("\n3. Iniciando K-Folds Cross Validation (K=5)...")
    pliegues = generar_k_folds(X_completo, y_completo, k=5)
    
    lista_accuracy = []
    lista_macro_f1 = []

    for i in range(5):
        print(f"   -> Evaluando Fold {i+1}/5...")
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
        lista_accuracy.append(metricas['Accuracy'])
        lista_macro_f1.append(metricas['Macro F1'])

    accuracy_promedio = sum(lista_accuracy) / len(lista_accuracy)
    macro_f1_promedio = sum(lista_macro_f1) / len(lista_macro_f1)
    
    print("\n=== Resultados de Evaluación (Promedio 5 Folds) ===")
    print(f"Accuracy Global: {accuracy_promedio:.4f}")
    print(f"Macro F1-Score:  {macro_f1_promedio:.4f}")

    # ==========================================
    # FASE DE PRODUCCIÓN: Entrenamiento Final
    # ==========================================
    print("\n4. Entrenando modelo final con el 100% de los datos...")
    modelo_final = NaiveBayesMesaAyuda()
    modelo_final.entrenar(X_completo, y_completo)

    ruta_modelo = 'modelo_guardado/modelo_nb.pkl'
    modelo_final.guardar_modelo(ruta_modelo)
    print(f"Modelo guardado exitosamente en: {ruta_modelo}")
    print("¡Backend completado! Listo para integrar con la interfaz web.")

if __name__ == '__main__':
    main()