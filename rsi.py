import os
import warnings
import pandas as pd
import numpy as np
from multiprocessing import Pool

warnings.filterwarnings("ignore")

# Define o número de processos a serem usados pelo multiprocessing
num_processes = os.cpu_count()

# Função para calcular o RSI de um DataFrame de preços
def calculate_rsi(prices, rsi_periods):
    deltas = np.diff(prices)
    seed = deltas[:rsi_periods+1]
    up = seed[seed >= 0].sum()//rsi_periods # converter resultado para inteiros usando //
    down = -seed[seed < 0].sum()//rsi_periods # converter resultado para inteiros usando //
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:rsi_periods] = 100. - 100./(1.+rs)

    for i in range(rsi_periods, len(prices)):
        delta = deltas[i-1]

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(rsi_periods-1) + upval)/rsi_periods
        down = (down*(rsi_periods-1) + downval)/rsi_periods

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


def calculate_moving_average_crossover(df, ma_short, ma_long, rsi_periods):
    # Calcula o RSI
    df['rsi'] = calculate_rsi(df['close'], rsi_periods)

    # Calcula as médias móveis do RSI
    col_short = 'ma_{}'.format(ma_short)
    col_long = 'ma_{}'.format(ma_long)
    df[col_short] = df['rsi'].rolling(ma_short).mean()
    df[col_long] = df['rsi'].rolling(ma_long).mean()

    # Identifica os cruzamentos de médias móveis
    df.loc[df[col_short] > df[col_long], 'cross'] = 1
    df.loc[df[col_short] <= df[col_long], 'cross'] = 0
    df['signal'] = df['cross'].diff()

    total_signals = len(df['signal'])
    total_correct_signals = sum(1 for s in df['signal'] if s > 0)

    efficiency_gain = 0 if total_signals == 0 else total_correct_signals / total_signals

    return efficiency_gain



# Função para processar um arquivo CSV e retornar as informações de ganho de eficiência
def process_csv_file(file_path, ma_short, ma_long, rsi_periods):
    df = pd.read_csv(file_path)
    efficiency_gain = calculate_moving_average_crossover(df, ma_short, ma_long, rsi_periods)
    return efficiency_gain


if __name__ == '__main__':
    # Define os períodos das médias móveis e o período do RSI
    ma_short = 30
    ma_long = 40
    rsi_periods = 15  # Defina este valor como quiser
    
    # Lista todos os arquivos CSV na pasta "data"
    csv_files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]

    # Processa os arquivos CSV usando multiprocessing
    with Pool(num_processes) as p:
        results = [p.apply_async(process_csv_file, args=(f, ma_short, ma_long, rsi_periods)) for f in csv_files]
        efficiency_gains = [(file_path, result.get()) for file_path, result in zip(csv_files, results)]

    # Exibe os resultados
    print("\nGanho de eficiência para cada arquivo:\n")
    for file_path, efficiency_gain in efficiency_gains:
        print("{}: {:.2}".format(os.path.basename(file_path), efficiency_gain))

    # Define outros períodos das médias móveis e do RSI
    
    print(f'\nAnálise feita com outras configurações de médias móveis e outro período do RSI')
    ma_short_2 = 60
    ma_long_2 = 80
    rsi_periods_2 = 20
    
    # Processa os arquivos CSV usando multiprocessing com outras médias móveis e o outro período do RSI
    with Pool(num_processes) as p:
        results = [p.apply_async(process_csv_file, args=(f, ma_short_2, ma_long_2, rsi_periods_2)) for f in csv_files]
        efficiency_gains_2 = [(file_path, result.get()) for file_path, result in zip(csv_files, results)]


    print("Ganho de eficiência para cada arquivo:")
    for file_path, efficiency_gain in efficiency_gains_2:
        print("{}: {:.2}".format(os.path.basename(file_path), efficiency_gain))