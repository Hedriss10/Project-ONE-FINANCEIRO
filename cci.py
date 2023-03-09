import os
import pandas as pd
import numpy as np
from multiprocessing import Pool

# Define o número de processos a serem usados pelo multiprocessing
num_processes = os.cpu_count()


def calculate_cci(prices, n=20):
    """Função para calcular o indicador Commodity Channel Index (CCI)"""
    tp = (prices["high"] + prices["low"] + prices["close"]) / 3
    ma = tp.rolling(n).mean()
    mdev = (tp - ma).abs().rolling(n).sum() / n
    cci = (tp - ma) / (0.015 * mdev)
    return cci


def calculate_moving_average_crossover(df, period_fast=10, period_slow=50):
    """Função para calcular um cruzamento de médias móveis usando o CCI como filtro"""
    # Calcula o CCI
    df['cci'] = calculate_cci(df)

    # Calcula as médias móveis com base no fechamento
    df[f"ma_{period_fast}"] = df["close"].rolling(period_fast).mean()
    df[f"ma_{period_slow}"] = df["close"].rolling(period_slow).mean()

    # Identifica os cruzamentos de médias móveis filtrados pelo CCI
    df.loc[((df["ma_" + str(period_fast)] > df["ma_" + str(period_slow)]) & (df["cci"] > 100)), "signal"] = 1
    df.loc[((df["ma_" + str(period_fast)] < df["ma_" + str(period_slow)]) & (df["cci"] < -100)), "signal"] = -1

    # Calcula as mudanças do sinal para contabilizar novas entradas
    df["signal"] = df["signal"].diff()

    # Exclui linhas sem sinais
    df.dropna(subset=["signal"], inplace=True)
    return df


# Função para processar um arquivo CSV e retornar as informações de cruzamento de médias móveis
def process_csv_file(file_path, period_fast=10, period_slow=50):
    df = pd.read_csv(file_path)
    df = calculate_moving_average_crossover(df, period_fast, period_slow)
    return df["signal"].values.sum()


if __name__ == '__main__':
    # Pergunta ao usuário quais são os períodos das médias móveis
    period_fast = int(input("Qual é o período da média móvel rápida? "))
    period_slow = int(input("Qual é o período da média móvel lenta? "))

    # Lista todos os arquivos CSV na pasta "data"
    csv_files = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.csv')]

    # Processa os arquivos CSV usando multiprocessing
    with Pool(num_processes) as p:
        results = [p.apply_async(process_csv_file, args=(f, period_fast, period_slow)) for f in csv_files]
        signals = [result.get() for result in results]

    # Calcula os ganhos de eficiência
    total_signals = len(signals)
    total_correct_signals = sum(1 for s in signals if s > 0)
    total_incorrect_signals = sum(1 for s in signals if s < 0)
    efficiency_gain = total_correct_signals / total_signals if total_signals > 0 else 0

    # Exibe os resultados
    print(f"Ganho de eficiência: {efficiency_gain:.2}")
    print(f"Número total de sinais: {total_signals}")
    print(f"Sinais corretos: {total_correct_signals}")
    print(f"Sinais incorretos: {total_incorrect_signals}")
