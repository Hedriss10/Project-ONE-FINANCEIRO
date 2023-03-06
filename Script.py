import os
import pandas as pd
import click
import retrying
import os
import glob
import pandas as pd
from multiprocessing import Pool

@retrying.retry(wait_fixed=1000, stop_max_delay=30000)
def ask_for_input(prompt):
    return click.prompt(prompt, type=str)


def process_data(file_path: str) -> None:
    df = pd.read_csv(file_path)

    print(f"Nome arquivo: {os.path.basename(file_path)}")
    print(f"Tamanho do dataframe: {len(df)}")
    table_name = ask_for_input("Digite o nome da tabela (open, high, low ou close): ")
    period = int(ask_for_input("Digite o valor do período: "))

    delta = df[table_name].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.shift(1)
    rsi = 100 - (100 / (1 + rs))
    
    # Atribuindo o resultado à coluna rsi:
    df['rsi'] = rsi

    total_gain = gain.sum()
    total_loss = loss.sum()

    print(f"Total de ganhos: {total_gain}")
    print(f"Total de perdas: {total_loss}")

if __name__ == '__main__':
    folder = os.path.join(os.getcwd(), 'data')
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

    with Pool() as pool:
        pool.map(process_data, files)
