import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

#Descarga de recursos necesarios para el procesamiento de texto
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def limpiar_texto(texto):
    """
    Comvierte el texto a minúsculas, elimina caracteres especiales y números, y tokeniza el texto en palabras individuales.
    """

    #Manejo de posibles valores nulos en el texto
    if not isinstance(texto, str):
        return ""
    
    #La expresión r'\{\{.*?\}\}' busca todo lo que esté entre dos llaves dobles y lo borra.
    texto = re.sub(r'\{\{.*?\}\}', ' ', texto)

    #Convertir a minúsculas
    texto = texto.lower()

    #Eliminar caracteres especiales y números
    texto = re.sub(r'[^a-z]+', ' ', texto)

    #Eliminar espacios adicionales
    texto = texto.strip()
    return texto

def preprocesar_ticket(texto, idioma='english'):
    """
    Se aplica un pipeline de NLP para transformar el ticket de texto
    en una lista de palabras procesadas
    """

    #Limpiar el texto
    texto_limpio = limpiar_texto(texto)

    if not texto_limpio:
        return []
    
    #Tokenizar el texto
    tokens = word_tokenize(texto_limpio, language=idioma)

    #Eliminar stopwords
    stop_words = set(stopwords.words(idioma))
    tokens_sin_stopwords = [palabra for palabra in tokens if palabra not in stop_words]

    #Aplicar stemming
    stemmer = SnowballStemmer(idioma)
    tokens_procesados = [stemmer.stem(palabra) for palabra in tokens_sin_stopwords]


    return tokens_sin_stopwords