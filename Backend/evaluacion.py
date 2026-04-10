import random

def generar_k_folds(X, y, k=5):
    """
    Divide los datos en k pliegues aleatorios para validación cruzada.
    Retorna una listas de pliegues, donde cada pliegue es una tupla (X_train, y_train, X_test, y_test).
    """

    #Emparejar los textos con sus etiquetas y barajarlos
    datos_combinados = list(zip(X, y))
    random.shuffle(datos_combinados)

    tamaño_pliegue = len(datos_combinados) // k
    pliegues = []

    for i in range(k):
        inicio = i * tamaño_pliegue
        #Si es el último pliegue, tomando en cuenta el resto para no perder datos
        fin = (i + 1) * tamaño_pliegue if i != k - 1 else len(datos_combinados)
        pliegues_actual = datos_combinados[inicio:fin]
        pliegues.append(pliegues_actual)
    return pliegues

def crear_matriz_confusion(y_reales, y_predichas, clases):
    """
    Contrucción de diccionario que representa la matriz nxn
    Filas: Valores reales | Columnas: Valores predichos
    """

    #Inicializar la matriz de confusión con ceros para las combinaciones de clases
    matriz = {clase_real: {clase_pred: 0 for clase_pred in clases} for clase_real in clases}

    for real, pred in zip(y_reales, y_predichas):
        matriz[real][pred] += 1
    return matriz

def calcular_metricas(y_reales, y_predichas, clases):
    """
    Calculo de métricas de la matriz de confusión
    """
    
    matriz = crear_matriz_confusion(y_reales, y_predichas, clases)
    metricas_por_clase = {}
    correctos_totales = 0
    total_instancia = len(y_reales)
    suma_f1 = 0

    for clase in clases:
        #Verdaderos positivos: El modelo predice C y realmente es C
        tp = matriz[clase][clase]
        correctos_totales += tp

        #Falsos Positivo: El modelo predijo C, pero en realidad era otra clase.
        fp = sum(matriz[otra_clase][clase] for otra_clase in clases if otra_clase != clase)

        # Falsos Negativos (FN): El modelo predijo otra clase, pero en realidad era C.
        fn = sum(matriz[clase][otra_clase] for otra_clase in clases if otra_clase != clase)

        #Calculos para la protección contra división por cero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.00
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.00
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.00

        metricas_por_clase[clase] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        }
        suma_f1 += f1_score

    # Accuracy Global: Proporción de predicciones correctas sobre el total de instancias.
    accuracy_global = correctos_totales / total_instancia if total_instancia > 0 else 0.00

    # F1-Score Macro: Promedio del F1-Score de cada clase.
    macro_f1 = suma_f1 / len(clases) if len(clases) > 0 else 0.00

    return{
        'Matriz de Confusión': matriz,
        'Métricas por Clase': metricas_por_clase,
        'Accuracy': accuracy_global,
        'Macro F1': macro_f1
    }
