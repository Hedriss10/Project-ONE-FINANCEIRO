"""
Na análise técnica de ações e outros ativos financeiros, 
a "P" usualmente se refere ao número de períodos de tempo usados para calcular uma média móvel e a "Q" se refere à quantidade de períodos de tempo usados para calcular uma segunda média móvel do resultado da primeira média móvel.

Uma estratégia comum é usar uma média móvel rápida para indicar tendências próximas ao que aconteceu no dia anterior e uma média móvel lenta relativamente, como o componente "suave" da tendência mais ampla do preço.
Portanto o P estabelece o número de períodos para se realizar a primeira média móvel e o Q o número distinto de período utilizado para suavizar a primeira média.

Returns:
    _type_: _description_
"""

#Validando test de timesleep
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from multiprocessing import Pool


# função para calcular as médias móveis do RSI
def calc_MM_RSI(periodo):
    # lê os dados
    df = pd.read_csv('data\\6A1.csv', parse_dates=['time'])
    df.set_index('time', inplace=True)

    # calcula o RSI com base no período fornecido pelo usuário
    rsi = RSIIndicator(df['close'], periodo).rsi()

    results = []

    # cria um loop para calcular todas as possibilidades de médias móveis do RSI
    for p in range(1, 21):
        for q in range(1, 21):
            # calcula a média móvel do RSI
            ma_rsi = rsi.rolling(window=p).mean().rolling(window=q).mean()

            # calcula o ganho de eficiência da média móvel do RSI
            ganho_eficiencia = ((ma_rsi.shift(-1) - df['close'].shift(-1)) / df['close'].shift(-1)).mean()

            # adiciona os resultados em uma lista
            results.append(f"Média móvel: {p}/{q} | Ganho de eficiência: {ganho_eficiencia:.2f}")

    return results

if __name__ == '__main__':
    # define o número de processos
    num_processes = 8

    # divide o período em partes iguais para cada processo
    periodos = [10] * num_processes

    with Pool(processes=num_processes) as pool:
        # executa a função calc_MM_RSI para cada processo e junta os resultados
        results = pool.map(calc_MM_RSI, periodos)

    # imprime a lista final de resultados 
    for sublist in results:
        for item in sublist:
            print(item)


