import math
import pickle
from collections import defaultdict

class NaiveBayesMesaAyuda:
    def __init__(self):
        #Probabilidad de cada clase
        self.log_prior = {}

        #Conteo de frecuencia de cada palabra por clase
        self.frecuencia_palabras = defaultdict(lambda: defaultdict(int))

        #Total de palabras sumads por cada clase
        self.total_palabras_clase = defaultdict(int)

        #Creación de vocabulario por cada clase
        self.vocabulario = set()
        self.tamano_vocabulario = 0
        self.clases = set()
    
    def entrenar(self, X_train, y_train):
        """
        Xtrain: Lista de listas de palabras
        y_train: Lista de etiquetas correspondientes a cada lista de palabras
        """
        total_documentos = len(y_train)
        conteo_clases = defaultdict(int)

        #Contar los documentos por clase y las frecuencias de palabras
        for i in range(total_documentos):
            documento = X_train[i]
            clase = y_train[i]

            self.clases.add(clase)
            conteo_clases[clase] += 1

            for palabra in documento:
                self.frecuencia_palabras[clase][palabra] += 1
                self.total_palabras_clase[clase] += 1
                self.vocabulario.add(palabra)
        
        self.tamano_vocabulario = len(self.vocabulario)

        #Calcular la probabilidad a priori logarítmica para cada clase
        for clase in self.clases:
            prob_priori = conteo_clases[clase] / total_documentos
            self.log_prior[clase] = math.log(prob_priori)
    
    def predecir_instancia(self, documento_tokenizado):
        """
        Predice la clase para un solo ticket ya procesado
        """
        scores_clases = self.calcular_scores(documento_tokenizado)
        
        #Retornar la clase con mayor score
        clase_ganadora = max(scores_clases, key=scores_clases.get)
        return clase_ganadora

    def calcular_scores(self, documento_tokenizado):
        """
        Calcula el puntaje logarítmico para cada clase.
        """
        scores_clases = {}

        for clase in self.clases:
            score = self.log_prior[clase]

            for palabra in documento_tokenizado:
                frecuencia = self.frecuencia_palabras[clase].get(palabra, 0)
                numerador = frecuencia + 1
                denominador = self.total_palabras_clase[clase] + self.tamano_vocabulario
                probabilidad_palabra = numerador / denominador

                score += math.log(probabilidad_palabra)

            scores_clases[clase] = score

        return scores_clases

    def predecir_proba(self, documento_tokenizado):
        """
        Devuelve probabilidades normalizadas por clase para una instancia.
        """
        scores = self.calcular_scores(documento_tokenizado)
        if not scores:
            return {}

        max_score = max(scores.values())
        exp_scores = {clase: math.exp(score - max_score) for clase, score in scores.items()}
        suma_scores = sum(exp_scores.values())

        if suma_scores == 0:
            return {clase: 0.0 for clase in scores}

        return {clase: valor / suma_scores for clase, valor in exp_scores.items()}
    
    def predecir(self, X_test):
        """
        Recive una lista de documentos y devuelve una lista de predicciones
        """
        predicciones = []
        for documento in X_test:
            predicciones.append(self.predecir_instancia(documento))
        return predicciones

    def guardar_modelo(self, ruta_archivo):
        """
        Guardar el modelo entrenado en un archivo
        """
        with open(ruta_archivo, 'wb') as archivo:
            pickle.dump({
                'log_prior': self.log_prior,
                'frecuencia_palabras': dict(self.frecuencia_palabras),
                'total_palabras_clase': dict(self.total_palabras_clase),
                'vocabulario': self.vocabulario,
                'tamano_vocabulario': self.tamano_vocabulario,
                'clases': self.clases
            }, archivo)
    
    def cargar_modelo(self, ruta_archivo):
        """
        Cargar un modelo previamente guardado desde un archivo
        """
        with open(ruta_archivo, 'rb') as archivo:
            datos = pickle.load(archivo)
            self.log_prior = datos['log_prior']
            self.frecuencia_palabras = defaultdict(
                lambda: defaultdict(int),
                {
                    clase: defaultdict(int, frecuencias)
                    for clase, frecuencias in datos['frecuencia_palabras'].items()
                },
            )
            self.total_palabras_clase = defaultdict(int, datos['total_palabras_clase'])
            self.vocabulario = set(datos.get('vocabulario', set()))
            self.tamano_vocabulario = datos['tamano_vocabulario']
            self.clases = set(datos['clases'])