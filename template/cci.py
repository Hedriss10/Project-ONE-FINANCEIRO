import os
import pandas as pd
import numpy as np
from multiprocessing import Pool


# Define o número de processos a serem usados pelo multiprocessing
num_processes = os.cpu_count()

# Função para calcular o CCI de um DataFrame de preços
def calculate_cci(prices, n=20):
    typical_prices = (prices['high'] + prices['low'] + prices['close']) / 3.0
    sma = typical_prices.rolling(n).mean()
    mad = abs(typical_prices - sma).rolling(n).mean()
    return (typical_prices - sma) / (0.015 * mad)


# Função para calcular as médias móveis do CCI
def calculate_moving_averages(df, ma_periods):
    for period in ma_periods:
        col_name = 'ma_{}'.format(period)
        df[col_name] = df['cci'].rolling(period).mean()


# Função para identificar os cruzamentos de médias móveis
def identify_crossovers(df, ma_periods):
    for i in range(1, len(ma_periods)):
        ma_short = ma_periods[i-1]
        ma_long = ma_periods[i]
        col_short = 'ma_{}'.format(ma_short)
        col_long = 'ma_{}'.format(ma_long)
        df.loc[df[col_short] > df[col_long], 'cross'] = 1
        df.loc[df[col_short] <= df[col_long], 'cross'] = 0
        df['signal'] = df['cross'].diff()


# Função para processar um arquivo CSV e retornar as informações de cruzamento de médias móveis
def process_csv_file(file_path, ma_periods):
    df = pd.read_csv(file_path)
    df['cci'] = calculate_cci(df, n=20)
    calculate_moving_averages(df, ma_periods)
    identify_crossovers(df, ma_periods)
    crossovers = [(ma_periods[i-1], ma_periods[i], df['signal'].sum()) for i in range(1, len(ma_periods))]
    return crossovers


if __name__ == '__main__':
    # Pergunta ao usuário quais são os períodos das médias móveis (separados por vírgula)
    ma_periods_str = input("Quais são os períodos das médias móveis (separados por vírgula)? ")
    ma_periods = [int(x.strip()) for x in ma_periods_str.split(',')]

    # Lista todos os arquivos CSV na pasta "data"
    csv_files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]

    # Processa os arquivos CSV usando multiprocessing
    with Pool(num_processes) as p:
        results = [p.apply_async(process_csv_file, args=(f, ma_periods)) for f in csv_files]
        crossovers = [result.get() for result in results]

    # Calcula o ganho de eficiência de cada combinação de médias móveis
    efficiency_gains = {}
    for i, ma_short in enumerate(ma_periods):
        for j, ma_long in enumerate(ma_periods):
            if ma_short >= ma_long:
                continue
            signals = []
            for file_crossovers in crossovers:
                for crossover in file_crossovers:
                    if crossover[0] == ma_short and crossover[1] == ma_long:
                        signals.append(crossover[2])
            if len(signals) > 0:
                total_signals = len(signals)
                total_correct_signals = sum(1 for s in signals if s > 0)
                efficiency_gains[(ma_short, ma_long)] = total_correct_signals / total_signals

    # Exibe os resultados
    print("Ganho de eficiência para cada combinação de médias móveis:")
    for ma_short, ma_long in efficiency_gains:
        print("  {}-{}: {:.2}".format(ma_short, ma_long, efficiency_gains[(ma_short, ma_long)]))

    print("\nCruzamentos de médias móveis para cada arquivo:")
    for i, file_path in enumerate(csv_files):
        print("\nArquivo {}: ".format(os.path.basename(file_path)))
        for j, crossover in enumerate(crossovers[i]):
            ma_short, ma_long, signal = crossover
            print("  {}-{}: {}".format(ma_short, ma_long, signal))
