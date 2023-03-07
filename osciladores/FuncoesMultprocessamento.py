import numpy as np
import concurrent.futures
from multiprocessing import cpu_count


# Renponsavel por dividir um intervalo escolhido para um numero de processos escolhido.
def num_execucoes(inicio:int, fim:int, step:int) -> int:
    x = (fim - inicio)//step
    if  (fim - inicio)%step:
        return x + 1
    return x

def lista_interv(inicio, fim, step, qtd):
	n = num_execucoes(inicio, fim, step)
	div = n//qtd
	arr = np.empty(qtd  + 1, dtype = np.int32)
	arr[:len(arr) - 1] = inicio + np.arange(len(arr) - 1)*div*step
	arr[len(arr) - 1] = fim
	# Divide igualmente o resto
	if qtd > 2:
		resto = n%qtd
		pesos = np.arange(1, resto)
		arr[len(arr) - resto:len(arr) - 1] += pesos*step
	return arr


def main(iteraveis:list[list]) -> list[list]:
	"""Recebe uma lista contendo varios iteraveis. Considera que é feito o
 produto cartesiano entre cada um dos elementos, retorna todas as combinações.

	Args:
		iteraveis (list[list]): Lista contendo listas com os elementos de
  cada loop.

	Returns:
		list[list]: Produto cartesiano entre as combinações possíveis.
	"""
	combinacao_dos_lacos = [None for _ in range(len(iteraveis))]
	todas_combinacao_dos_lacos = []
 
	def nested_loops(iteraveis:list[list], i):
		if i < len(iteraveis):
			for x in iteraveis[i]:
				combinacao_dos_lacos[i] = x
				nested_loops(iteraveis, i + 1)
		else:
			todas_combinacao_dos_lacos.append(combinacao_dos_lacos.copy())

	nested_loops(iteraveis, 0)
	return todas_combinacao_dos_lacos


def divide_intervalos(iteraveis:list[list], repeticoes:int,
            	qtd_intervalos:int) -> list[list[list[int]]]:
	indices_inicio_e_fim = lista_interv(0, repeticoes, 1, qtd_intervalos)
	
	# Faz isso para pegar somente os indices
	todas_combinacoes = main(
     [range(len(iteravel)) for iteravel in iteraveis])
	
	combinacoes_dividas_interavalos = [None for _ in range(qtd_intervalos)]
	for i, (inicio, fim) in enumerate(zip(indices_inicio_e_fim[:-1], indices_inicio_e_fim[1:])):
		combinacoes_dividas_interavalos[i] = todas_combinacoes[inicio:fim]
	return combinacoes_dividas_interavalos

def get_qtd_process(repeticoes:int) -> int:
    return min(cpu_count(), repeticoes)

def divide_intervalos_antigo(inicio, fim, step, qtd_intervalos):
	interv = lista_interv(inicio, fim, step, qtd_intervalos)
    # Cada intervalo vai ter 3 parametros: inicio, fim e step
	intervalos_dividos = np.empty((qtd_intervalos, 3), dtype = np.int32)
	for i in range(1, len(interv)):
		intervalos_dividos[i - 1] = interv[i - 1], interv[i], step
	return intervalos_dividos

def get_qtd_reps_por_processo(repeticoes:int) -> list[int]:
    qtd_process = get_qtd_process(repeticoes)
    intervalos = divide_intervalos_antigo(0, repeticoes, 1, qtd_process)
    
    qtd_reps_por_processo = []
    for inicio, fim, step in intervalos:
        qtd_reps_por_processo.append(num_execucoes(inicio, fim, step))
    return qtd_reps_por_processo

def get_intervalos_por_processo(iteravel:list) -> list[list]:
    qtd_process = get_qtd_process(len(iteravel))
    return divide_intervalos_antigo(0, len(iteravel), 1, qtd_process)

def divide_lacos_aninhados_por_processo(*iteraveis:list[list]) -> list[list[list[int]]]:
	"""Divide entre diferentes processos as execuções de múltiplos
	loops aninhados. Para isso, recebe os argumentos de cada loop a
	a ser particionado.

	Returns:
		list[list[list[int]]]: Uma lista para cada processo. Dentro de cada
	lista por processo, há uma lista contendo os indices de cada loop a serem
	executados.

	Ex:
	Processo 0
	for indices_lacos_particionados in args_lacos_divididos_por_processo[0]:
		i_laco_0 = indices_lacos_particionados[0]
		i_laco_1 = indices_lacos_particionados[1]
  		...
	"""
	repeticoes = 1
	for interavel in iteraveis:
		repeticoes *= len(interavel)

	qtd_process = get_qtd_process(repeticoes)
	return divide_intervalos(iteraveis, repeticoes, qtd_process)

