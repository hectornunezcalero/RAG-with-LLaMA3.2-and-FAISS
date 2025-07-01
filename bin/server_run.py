# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Universidad de Alcalá - Escuela Politécnica Superior            #
#                                                                       #
#       Grado en Ingeniería Telemática - Curso 2025/2026                #
#                                                                       #
#                                                                       #
#       Trabajo de Fin de Grado:                                        #
#           Sistema de Generación Aumentada por Recuperación (RAG)      #
#           con LLaMA 3.2 como LLM para consultas                       #
#           sobre documentos o artículos en PDF                         #
#                                                                       #
#                                                                       #
#       Autor: Héctor Núñez Calero                                      #
#       Tutor: Jorge Pérez Aracil                                       #
#       Cotutor: Alberto Palomo Alonso                                  #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#       Script: server_run.py                                           #
#       Funciones principales:                                          #
#        1. Inicializar la Clase perteneciente al servidor              #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #

from src import Llama3Server

if __name__ == "__main__":
    print("Iniciando servidor con LLaMA 3.2...")
    Llama3Server()

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                                       #
#                               END OF FILE                             #
#                                                                       #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #