import pandas as pd 
import os
import sys 

#Accc..
sys.path.append("..")



#Fazendo a requisição dos dados da na pasta data
class RequestDados:
    #Criando a função para fazer a manipulação dos nomes do arquivos. 
    def RequestData(self):
    #Pecorrendo até a pasta data aonde está todos os datasets
        folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path =  os.path.join(folder, 'data')

        # Percorre todos os arquivos na pasta
        for file_name in os.listdir(path):

            # Verifica se a string 'download' está presente no nome do arquivo
            if 'download' in file_name:
                
                # Define o novo nome do arquivo sem a string 'download'
                new_file_name = file_name.replace(' ', '').replace("download", '')
                
                # Utiliza o método rename para renomear o arquivo
                os.rename(os.path.join(path, file_name), os.path.join(path, new_file_name))
