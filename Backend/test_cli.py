from naive_bayes import NaiveBayesMesaAyuda
from preprocesamiento import preprocesar_ticket

def main():
    print("=== Consola de Pruebas: Clasificador de Tickets ===")
    print("Cargando modelo...")
    
    # Instanciamos un modelo vacío y lo llenamos con los datos guardados
    modelo = NaiveBayesMesaAyuda()
    ruta_modelo = 'modelo_guardado/modelo_nb.pkl'
    
    try:
        modelo.cargar_modelo(ruta_modelo)
        print("¡Modelo cargado con éxito!")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_modelo}.")
        return

    print("-" * 50)
    print("Escribe un ticket de prueba (o escribe 'salir' para terminar).")
    print("-" * 50)

    while True:
        # Recibimos el texto del usuario desde la terminal
        texto_usuario = input("\nIngresa el problema del cliente: ")
        
        if texto_usuario.lower() in ['salir', 'exit', 'quit']:
            print("Cerrando entorno de pruebas...")
            break
            
        if not texto_usuario.strip():
            continue

        # 1. Pasamos el texto crudo por el mismo preprocesamiento del entrenamiento
        tokens_limpios = preprocesar_ticket(texto_usuario, idioma='english')
        
        # 2. Le pedimos al modelo que prediga la categoría usando la suma de logaritmos
        prediccion = modelo.predecir_instancia(tokens_limpios)
        
        print(f"\n>> Tokens procesados: {tokens_limpios}")
        print(f">> Categoría Predicha: **{prediccion}**")

if __name__ == '__main__':
    main()