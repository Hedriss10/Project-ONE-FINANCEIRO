import glob
import os
import pandas as pd
import pandas_ta as ta

from multiprocessing import Pool

#Calculando o rsi
def calculate_rsi(filename, periods, price_column, show_names=False):
    #Carregamento de dados ao path
    df = pd.read_csv(filename, parse_dates=True)
    delta = df[price_column].diff()
    #Gains e loss
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    #Declaração
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    #Indicadores
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    #atribuindo os gains loss e rsi 
    df['gain'] = gain 
    df['loss'] = loss
    df['rsi'] = rsi
        
    if df.shape[0] > 0:
        return rsi.iloc[-1], filename
    else:
        return None, filename


#Calculate do cci
def calculate_cci(filename, periods, price_column, constant, show_names=False):
    df = pd.read_csv(filename, parse_dates=True)
    
    typical_price = (df['high'] + df['low'] + df[price_column]) / 3.0
    moving_average = typical_price.rolling(window=periods).mean()
    mean_deviation = abs(typical_price - moving_average).rolling(window=periods).mean()
    cci = (typical_price - moving_average) / (constant * mean_deviation)
    
    # Atribuindo o CCI ao dataframe
    df['cci'] = cci
    
    if df.shape[0] > 0:
        return cci.iloc[-1], filename
    else:
        return None, filename



if __name__ == '__main__':
    # Entrada de período, coluna de preço e constante do CCI
    periods = int(input("Digite o número de períodos para o rsi e cci: "))
    price_column = input("Digite o nome da coluna de preço: ").lower()
    constant = float(input("Digite a constante do CCI: "))

    # Selecionar arquivos CSV da pasta data
    csv_files = glob.glob('data/*.csv')

    # Verifica se deseja processar todos os arquivos
    process_all = input(f"Processar todos os arquivos? (S/N)").lower() == "s"

    # Seleção manual de arquivo
    if not process_all:
        # Exibe a lista de arquivos CSV
        print("\nSelecione o arquivo para processar:")
        for i, filename in enumerate(csv_files):
            print(f"{i+1}. {filename}")

        # Obtém o arquivo selecionado
        while True:
            try:
                file_index = int(input("> "))
                filename = csv_files[file_index-1]                
                break
            except:
                print("Índice inválido. Tente novamente.")

        # Processa o arquivo selecionado
        rsi, cci = calculate_rsi(filename, periods, price_column), calculate_cci(filename, periods, price_column, constant)
        if rsi[0] and cci[0]:
            print(f"\nResultado para {os.path.basename(filename)}:")
            print(f"RSI de {rsi[0]:.2f}")
            print(f"CCI de {cci[0]:.2f}")
        else:
            print(f"\nArquivo {os.path.basename(filename)} vazio ou inexistente")
         # Processa todos os arquivos
   
    else:
        # Cria um pool de processos com o número de CPUs disponíveis
        num_cpus = os.cpu_count()
        with Pool(num_cpus) as p:
            # Executa a função calculate_rsi e calculate_cci para cada arquivo CSV em paralelo
            results = p.starmap(calculate_rsi, [(filename, periods, price_column) for filename in csv_files])
            cci_results = p.starmap(calculate_cci, [(filename, periods, price_column, constant) for filename in csv_files])
            p.close()
            p.join()

        # Obtém o melhor resultado do RSI
        best_rs = None
        best_rsi = None
        best_filename = None
        for result, filename in results:
            if result and (best_rsi is None or result > best_rsi):
                best_rs = os.path.basename(filename)
                best_rsi = result
                best_filename = filename

        # Imprime o melhor resultado
        if best_rsi:
            print(f"\nMelhor resultado para {best_rs}:")
            print(f"RSI de {best_rsi:.2f}")
        else:
            print("Nenhum arquivo CSV válido encontrado")



        # Obtém o melhor resultado do CCI
        best_cci = None
        best_cci_val = None
        best_cci_filename = None
        for result, filename in results:
            if result and (best_cci_val is None or abs(result) > abs(best_cci_val)):
                best_cci = os.path.basename(filename)
                best_cci_val = result
                best_cci_filename = filename

        # Imprime o melhor resultado
        if best_cci:
            print(f"\nMelhor resultado para {best_cci}:")
            print(f"CCI de {best_cci_val:.2f}")
        else:
            print("Nenhum arquivo CSV válido encontrado")