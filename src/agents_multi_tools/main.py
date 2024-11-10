# main.py
import os
from dotenv import load_dotenv
import sys
from crewLoader import DocumentLoaderCrew
from crewSearcher import DocumentSearcherCrew

def clear_screen():
    """Limpia la pantalla de la consola"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_menu():
    """Imprime el menú principal"""
    clear_screen()
    print("\n=== SISTEMA DE GESTIÓN DOCUMENTAL ===")
    print("1. Cargar archivo")
    print("2. Consultar Base de Datos")
    print("3. Salir")
    print("=====================================")

def cargar_archivo(crew_loader):
    """Gestiona la carga de archivos"""
    clear_screen()
    print("\n=== CARGAR ARCHIVO ===")
    file_path = input("\nPor favor ingrese la ruta del archivo a cargar: ")
    
    if not os.path.exists(file_path):
        print("\nError: El archivo no existe!")
        input("\nPresione Enter para continuar...")
        return
    
    try:
        result = crew_loader.process_file(file_path)
        print("\nProceso completado. Resultado:")
        print(result)
    except Exception as e:
        print(f"\nError durante el procesamiento: {str(e)}")
    
    input("\nPresione Enter para continuar...")

def consultar_bd(crew_searcher):
    """Gestiona las consultas a la base de datos"""
    clear_screen()
    print("\n=== CONSULTAR BASE DE DATOS ===")
    prompt = input("\nIngrese su consulta: ")
    
    try:
        result = crew_searcher.search(prompt)
        print("\nResultado de la consulta:")
        print(result)
    except Exception as e:
        print(f"\nError durante la consulta: {str(e)}")
    
    input("\nPresione Enter para continuar...")

def main():
    # Configuración inicial
    load_dotenv()
    
    # Inicializar los Crews
    crew_loader = DocumentLoaderCrew()
    crew_searcher = DocumentSearcherCrew()
    
    while True:
        print_menu()
        
        try:
            opcion = input("\nSeleccione una opción (1-3): ")
            
            if opcion == "1":
                cargar_archivo(crew_loader)
            elif opcion == "2":
                consultar_bd(crew_searcher)
            elif opcion == "3":
                clear_screen()
                print("\nGracias por usar el sistema. ¡Hasta pronto!")
                sys.exit(0)
            else:
                print("\nOpción no válida. Por favor, intente nuevamente.")
                input("\nPresione Enter para continuar...")
        
        except Exception as e:
            print(f"\nError inesperado: {str(e)}")
            input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    main()