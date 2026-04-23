import random

def generar_k_folds(X, y, k=5, seed=42):
    """
    Divide los datos en k pliegues aleatorios para validación cruzada.
    Retorna una listas de pliegues, donde cada pliegue es una tupla (X_train, y_train, X_test, y_test).
    """

    if k <= 1:
        raise ValueError("k debe ser mayor que 1 para validacion cruzada")

    #Emparejar los textos con sus etiquetas y barajarlos
    datos_combinados = list(zip(X, y))
    random.Random(seed).shuffle(datos_combinados)

    pliegues = []
    total_datos = len(datos_combinados)
    tamano_base = total_datos // k
    residuo = total_datos % k
    inicio = 0

    for i in range(k):
        # Reparte el residuo entre los primeros pliegues para no perder instancias.
        tamano_actual = tamano_base + (1 if i < residuo else 0)
        fin = inicio + tamano_actual
        pliegues_actual = datos_combinados[inicio:fin]
        pliegues.append(pliegues_actual)
        inicio = fin

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


def inicializar_matriz_confusion(clases):
    return {clase_real: {clase_pred: 0 for clase_pred in clases} for clase_real in clases}


def acumular_matriz_confusion(matriz_acumulada, matriz_nueva, clases):
    for clase_real in clases:
        for clase_pred in clases:
            matriz_acumulada[clase_real][clase_pred] += matriz_nueva[clase_real][clase_pred]


def promedio_metricas_por_clase(metricas_por_fold, clases):
    acumulado = {
        clase: {'Precision': 0.0, 'Recall': 0.0, 'F1-Score': 0.0}
        for clase in clases
    }

    cantidad_folds = len(metricas_por_fold)
    if cantidad_folds == 0:
        return acumulado

    for metricas_fold in metricas_por_fold:
        for clase in clases:
            valores = metricas_fold.get(clase, {})
            acumulado[clase]['Precision'] += valores.get('Precision', 0.0)
            acumulado[clase]['Recall'] += valores.get('Recall', 0.0)
            acumulado[clase]['F1-Score'] += valores.get('F1-Score', 0.0)

    for clase in clases:
        acumulado[clase]['Precision'] /= cantidad_folds
        acumulado[clase]['Recall'] /= cantidad_folds
        acumulado[clase]['F1-Score'] /= cantidad_folds

    return acumulado
