from naive_bayes import NaiveBayesMesaAyuda
from preprocesamiento import preprocesar_ticket

def main():
    print("=== Consola de Pruebas: Clasificador de Tickets (11 Categorías) ===")
    print("Cargando modelo...")
    
    # Instanciamos un modelo vacío y lo llenamos con los datos guardados en .pkl
    modelo = NaiveBayesMesaAyuda()
    ruta_modelo = 'modelo_guardado/modelo_nb.pkl'
    
    try:
        modelo.cargar_modelo(ruta_modelo)
        print("¡Modelo cargado con éxito!")
        print(f"Categorías en memoria: {modelo.clases}")
    except FileNotFoundError:
        print(f"\nError: No se encontró el archivo {ruta_modelo}.")
        print("Asegúrate de haber ejecutado 'python train.py' primero para generar el modelo.")
        return

    print("\n" + "="*70)
    print("Escribe un ticket de prueba en inglés (o escribe 'salir' para terminar).")
    print("Ejemplo: 'I need to cancel my subscription before the next billing cycle'")
    print("="*70)

    while True:
        # Recibimos el texto del usuario desde la terminal
        texto_usuario = input("\nIngresa el problema del cliente: ")
        
        if texto_usuario.lower() in ['salir', 'exit', 'quit', 'q']:
            print("Cerrando entorno de pruebas. ¡Buen trabajo, ingeniero!")
            break
            
        if not texto_usuario.strip():
            continue

        # 1. Pasamos el texto crudo por tu función de preprocesamiento (regex, stopwords, stemming)
        tokens_limpios = preprocesar_ticket(texto_usuario, idioma='english')
        
        if not tokens_limpios:
            print("\n>> El texto ingresado no contiene palabras clave válidas después de limpiarlo.")
            continue
            
        # 2. Le pedimos al modelo que prediga la categoría usando la suma de logaritmos
        prediccion = modelo.predecir_instancia(tokens_limpios)
        
        print("-" * 50)
        print(f">> Tokens procesados (Stemming): {tokens_limpios}")
        print(f">> Categoría Predicha por la IA: **{prediccion}**")
        print("-" * 50)

if __name__ == '__main__':
    main()