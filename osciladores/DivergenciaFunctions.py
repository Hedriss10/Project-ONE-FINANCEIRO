from functools import cache
import numpy as np
import numba as nb
from Fontes import Fonte
from Indicadores import Indicador
from ElosDeRepeticao import MemCompartilhadaElos
from ParametrosDivergencias import COLUNAS_USADAS_POR_INDICADOR_ARQUIVO_SAIDA, QTD_GRUPOS_BARRAS_BACKTEST, ACERTO_MINIMO_TOP, ACERTO_MINIMO_BOT
from Divergencia import Divergencia
import pandas as pd


INDICACOES_TOP = {
    -1:"SHORT",
    0:"NEUTRO",
    1:"LONG"
}

INDICACOES_BOT = {
    -1:"LONG",
    0:"NEUTRO",
    1:"SHORT"
}

@nb.njit(cache = True)
def fez_diverg_bull_forte(fonte_da_divergencia:np.ndarray, indicador:np.ndarray, i_barra_hoje:int, i_barra:int) -> int:
    return 1 if (fonte_da_divergencia[i_barra_hoje] < fonte_da_divergencia[i_barra] and
                       indicador[i_barra_hoje] > indicador[i_barra]) else 0
@nb.njit(cache = True)
def fez_diverg_bull_media(fonte_da_divergencia:np.ndarray, indicador:np.ndarray, i_barra_hoje:int, i_barra:int) -> int:
    return 1 if (fonte_da_divergencia[i_barra_hoje] == fonte_da_divergencia[i_barra] and
                       indicador[i_barra_hoje] >  indicador[i_barra]) else 0
@nb.njit(cache = True)
def fez_diverg_bull_fraca(fonte_da_divergencia:np.ndarray, indicador:np.ndarray, i_barra_hoje:int, i_barra:int) -> int:
    return 1 if (fonte_da_divergencia[i_barra_hoje] <  fonte_da_divergencia[i_barra] and
                       indicador[i_barra_hoje] == indicador[i_barra]) else 0
@nb.njit(cache = True)
def fez_diverg_bull_escondida(fonte_da_divergencia:np.ndarray, indicador:np.ndarray, i_barra_hoje:int, i_barra:int) -> int:
    return 1 if (fonte_da_divergencia[i_barra_hoje] >  fonte_da_divergencia[i_barra] and
                       indicador[i_barra_hoje] < indicador[i_barra]) else 0

@nb.njit(cache = True)
def fez_diverg_bear_forte(fonte_da_divergencia:np.ndarray, indicador:np.ndarray, i_barra_hoje:int, i_barra:int) -> int:
    return -1 if (fonte_da_divergencia[i_barra_hoje] >  fonte_da_divergencia[i_barra] and
                       indicador[i_barra_hoje] < indicador[i_barra]) else 0
@nb.njit(cache = True)
def fez_diverg_bear_media(fonte_da_divergencia:np.ndarray, indicador:np.ndarray, i_barra_hoje:int, i_barra:int) -> int:
    return -1 if (fonte_da_divergencia[i_barra_hoje] == fonte_da_divergencia[i_barra] and
                       indicador[i_barra_hoje] <  indicador[i_barra]) else 0
@nb.njit(cache = True)
def fez_diverg_bear_fraca(fonte_da_divergencia:np.ndarray, indicador:np.ndarray, i_barra_hoje:int, i_barra:int) -> int:
    return -1 if (fonte_da_divergencia[i_barra_hoje] >  fonte_da_divergencia[i_barra] and
                       indicador[i_barra_hoje] == indicador[i_barra]) else 0
@nb.njit(cache = True)
def fez_diverg_bear_escondida(fonte_da_divergencia:np.ndarray, indicador:np.ndarray, i_barra_hoje:int, i_barra:int) -> int:
    return -1 if (fonte_da_divergencia[i_barra_hoje] < fonte_da_divergencia[i_barra] and
                       indicador[i_barra_hoje] > indicador[i_barra]) else 0



def busca_diverg_entre_indicador_e_fonte_da_diverg(fonte_da_divergencia:np.ndarray, indicador:np.ndarray, i_barra_hoje:int, i_barra:int):
    if fonte_da_divergencia[i_barra_hoje] >= fonte_da_divergencia[i_barra]:
        if indicador[i_barra_hoje] < indicador[i_barra]:
            return True
    else:
        if indicador[i_barra_hoje] >= indicador[i_barra]:
            return True
    return False

# def fonte_cresceu(fonte_da_divergencia:np.ndarray, i_barra:int) -> bool:
#     return fonte_da_divergencia[i_barra] >= fonte_da_divergencia[i_barra - 1]

# def formou_pico(fonte_da_divergencia:np.ndarray, i_barra:int) -> bool:
#     return fonte_da_divergencia[i_barra] >= fonte_da_divergencia[i_barra - 1] and fonte_da_divergencia[i_barra] >= fonte_da_divergencia[i_barra + 1]

# def fonte_decresceu(fonte_da_divergencia:np.ndarray, i_barra:int) -> bool:
#     return fonte_da_divergencia[i_barra] <= fonte_da_divergencia[i_barra - 1]

# def formou_vale(fonte_da_divergencia:np.ndarray, i_barra:int) -> bool:
#     return fonte_da_divergencia[i_barra] <= fonte_da_divergencia[i_barra - 1] and fonte_da_divergencia[i_barra] <= fonte_da_divergencia[i_barra + 1]

@nb.njit(cache = True)
def get_ocorrencias_divergencia_perfeita(sinais_gerados:np.ndarray, gabarito:np.ndarray,
    indices_gabarito_nao_nulo_por_grupo:np.ndarray, QTD_OCORRENCIAS_NEUTRAS:np.ndarray) -> tuple[list[int], list[int], bool]:
    # Se usar, retirar o comentario do Parametros Divergencia
    # "qtd_indices_gabarito_nulos_arr[0] = qtd_indices_gabarito_nulos[0]#  + 1"
    """Retorna o numero de divergencias ocorrida para uma divergencia perfeita. Esta
    eh qualificada por uma divergencia que só erra ou só acerta.
    Além do numero de ocorrências, retorna um bool que informa se a divergencia apenas
    acerta (True) ou apenas erra (False).
    Percorre as barras do dia mais recente para o dia mais antigo.

    Args:
        sinais_gerados (np.ndarray): Sinais gerados pela divergencia (1 Long, 0 Neutro e -1 Short)
        
        gabarito (np.ndarray): Movimento do mercado, 1 para alta, -1 para baixa e 0 para dia neutro.
        
        indices_gabarito_nao_nulo_por_grupo (np.ndarray): Guarda os indices do "gabarito" que podem ser percorridos.
        Quando não há variação no dia, não é possível determinar se é um dia long ou um dia short, por isso
        esses dias não são analisados

        QTD_OCORRENCIAS_NEUTRAS (np.ndarray): É o número de "gabaritos" neutros para cada grupo de barras analisado.
        Para tratar de dias sem variação, é empregada a abordagem que considera que a divergência permanece perfeita
        após o dia neutro. Por isso, uma divergencia que apenas erra considera que errou nesse dia, enquanto que uma
        que apenas acerta considera que acertou nesse dia.

    Returns:
        tuple[np.ndarray[int], list[int], bool]:
            [0] - Quantidade de ocorrencias por grupo
            [1] - Todos os indices de ocorrencia. Para encontrar as ocorrencias de um grupo só eh feito l[:ocorrencias]
            [2] - Divergencia eh apenas acerto ou eh apenas erro
    """
    ocorrencias = np.empty(QTD_GRUPOS_BARRAS_BACKTEST, np.int32)
    i_barras_com_ocorrencias = []
    qtd_ocorrencias = 0
    for i_grupo, indices_gabarito_nao_nulo in enumerate(indices_gabarito_nao_nulo_por_grupo):
        qtd_ocorrencias += QTD_OCORRENCIAS_NEUTRAS[i_grupo]
        i_indice = 0
        while i_indice < len(indices_gabarito_nao_nulo):
            i = indices_gabarito_nao_nulo[i_indice]
            if sinais_gerados[i]:
                if sinais_gerados[i] == gabarito[i]:
                    i_barras_com_ocorrencias.append(i)
                    i_indice += 1
                    get_ocorrencias_apenas_acertos(sinais_gerados,
                                        gabarito, indices_gabarito_nao_nulo, indices_gabarito_nao_nulo_por_grupo,
                                        QTD_OCORRENCIAS_NEUTRAS, qtd_ocorrencias, i_indice,
                                        ocorrencias, i_barras_com_ocorrencias, i_grupo)
                    return ocorrencias, i_barras_com_ocorrencias, True
                else:
                    i_barras_com_ocorrencias.append(i)
                    i_indice += 1
                    get_ocorrencias_apenas_erros(sinais_gerados,
                                        gabarito, indices_gabarito_nao_nulo, indices_gabarito_nao_nulo_por_grupo,
                                        QTD_OCORRENCIAS_NEUTRAS, qtd_ocorrencias, i_indice,
                                        ocorrencias, i_barras_com_ocorrencias, i_grupo)
                    return ocorrencias, i_barras_com_ocorrencias, False
            i_indice += 1
        ocorrencias[i_grupo] = 0
    ocorrencias[0] = -1 # Marca o primeiro com "-1", pois n houve sinais gerados
    return ocorrencias, i_barras_com_ocorrencias, True

@nb.njit(cache = True)
def get_ocorrencias_apenas_acertos(sinais_gerados:np.ndarray, gabarito:np.ndarray,
                                 indices_gabarito_nao_nulo:np.ndarray, indices_gabarito_nao_nulo_por_grupo:np.ndarray,
                                 QTD_OCORRENCIAS_NEUTRAS:list[int], qtd_ocorrencias:int,
                                 i_inicio:int, ocorrencias:np.ndarray, i_barras_com_ocorrencias:list[int],
                                 i_grupo:int) -> tuple[int, bool]:
    
    for i in indices_gabarito_nao_nulo[i_inicio:]:
        if sinais_gerados[i]:
            if sinais_gerados[i] == gabarito[i]:
                i_barras_com_ocorrencias.append(i)
                qtd_ocorrencias += 1
            else:
                ocorrencias[i_grupo] = -1
                return
    ocorrencias[i_grupo] = qtd_ocorrencias

    i_grupo += 1
    for i_grupo in range(i_grupo, len(indices_gabarito_nao_nulo_por_grupo)):
        qtd_ocorrencias += QTD_OCORRENCIAS_NEUTRAS[i_grupo]
        indices_gabarito_nao_nulo = indices_gabarito_nao_nulo_por_grupo[i_grupo]
        i_indice = 0
        while i_indice < len(indices_gabarito_nao_nulo):
            i = indices_gabarito_nao_nulo[i_indice]
            if sinais_gerados[i]:
                if sinais_gerados[i] == gabarito[i]:
                    i_barras_com_ocorrencias.append(i)
                    qtd_ocorrencias += 1
                else:
                    ocorrencias[i_grupo] = -1
                    return
            i_indice += 1
        ocorrencias[i_grupo] = qtd_ocorrencias


@nb.njit(cache = True)
def get_ocorrencias_apenas_erros(sinais_gerados:np.ndarray, gabarito:np.ndarray,
                                 indices_gabarito_nao_nulo:np.ndarray, indices_gabarito_nao_nulo_por_grupo:np.ndarray,
                                 QTD_OCORRENCIAS_NEUTRAS:list[int], qtd_ocorrencias:int,
                                 i_inicio:int, ocorrencias:np.ndarray, i_barras_com_ocorrencias:list[int],
                                 i_grupo:int) -> tuple[int, bool]:

    for i in indices_gabarito_nao_nulo[i_inicio:]:
        if sinais_gerados[i]:
            if sinais_gerados[i] != gabarito[i]:
                i_barras_com_ocorrencias.append(i)
                qtd_ocorrencias += 1
            else:
                ocorrencias[i_grupo] = -1
                return
    ocorrencias[i_grupo] = qtd_ocorrencias

    i_grupo += 1
    for i_grupo in range(i_grupo, len(indices_gabarito_nao_nulo_por_grupo)):
        qtd_ocorrencias += QTD_OCORRENCIAS_NEUTRAS[i_grupo]
        indices_gabarito_nao_nulo = indices_gabarito_nao_nulo_por_grupo[i_grupo]
        i_indice = 0
        while i_indice < len(indices_gabarito_nao_nulo):
            i = indices_gabarito_nao_nulo[i_indice]
            if sinais_gerados[i]:
                if sinais_gerados[i] != gabarito[i]:
                    i_barras_com_ocorrencias.append(i)
                    qtd_ocorrencias += 1
                else:
                    ocorrencias[i_grupo] = -1
                    return
            i_indice += 1
        ocorrencias[i_grupo] = qtd_ocorrencias

    

@nb.njit(cache = True)
def get_ocorrencias_divergencia(sinais_gerados:np.ndarray, gabarito:np.ndarray,
    indices_gabarito_nao_nulo_por_grupo:np.ndarray, QTD_OCORRENCIAS_NEUTRAS:np.ndarray) -> tuple[list[int], list[int], bool]:
    ocorrencias = np.empty(QTD_GRUPOS_BARRAS_BACKTEST, np.int32)
    acertos = np.empty(QTD_GRUPOS_BARRAS_BACKTEST, np.int32)

    i_barras_com_ocorrencias = []
    qtd_acertos = 0
    qtd_erros = 0
    qtd_ocorrencias = 0
    for i_grupo, indices_gabarito_nao_nulo in enumerate(indices_gabarito_nao_nulo_por_grupo):
        qtd_ocorrencias += QTD_OCORRENCIAS_NEUTRAS[i_grupo]
        i_indice = 0
        while i_indice < len(indices_gabarito_nao_nulo):
            i = indices_gabarito_nao_nulo[i_indice]
            if sinais_gerados[i]:
                if sinais_gerados[i] == gabarito[i]:
                    i_barras_com_ocorrencias.append(i)
                    i_indice += 1
                    qtd_acertos += 1
                    get_ocorrencias_deu_sinal(sinais_gerados,
                                        gabarito, indices_gabarito_nao_nulo, indices_gabarito_nao_nulo_por_grupo,
                                        QTD_OCORRENCIAS_NEUTRAS, qtd_ocorrencias, qtd_acertos, qtd_erros, i_indice,
                                        ocorrencias, i_barras_com_ocorrencias, acertos, i_grupo)
                    return ocorrencias, acertos, i_barras_com_ocorrencias, True
                else:
                    i_barras_com_ocorrencias.append(i)
                    i_indice += 1
                    qtd_erros += 1
                    get_ocorrencias_deu_sinal(sinais_gerados,
                                        gabarito, indices_gabarito_nao_nulo, indices_gabarito_nao_nulo_por_grupo,
                                        QTD_OCORRENCIAS_NEUTRAS, qtd_ocorrencias, qtd_acertos, qtd_erros, i_indice,
                                        ocorrencias, i_barras_com_ocorrencias, acertos, i_grupo)
                    return ocorrencias, acertos, i_barras_com_ocorrencias, False
            i_indice += 1
        ocorrencias[i_grupo] = 0
    ocorrencias[0] = -1 # Marca o primeiro com "-1", pois n houve sinais gerados
    return ocorrencias, acertos, i_barras_com_ocorrencias, True

@nb.njit(cache = True)
def get_ocorrencias_deu_sinal(sinais_gerados:np.ndarray, gabarito:np.ndarray,
                                 indices_gabarito_nao_nulo:np.ndarray, indices_gabarito_nao_nulo_por_grupo:np.ndarray,
                                 QTD_OCORRENCIAS_NEUTRAS:list[int], qtd_ocorrencias:int, qtd_acertos:int, qtd_erros:int,
                                 i_inicio:int, ocorrencias:np.ndarray, i_barras_com_ocorrencias:list[int],
                                 acertos:np.ndarray, i_grupo:int) -> tuple[int, bool]:
    
    for i in indices_gabarito_nao_nulo[i_inicio:]:
        if sinais_gerados[i]:
            i_barras_com_ocorrencias.append(i)
            qtd_ocorrencias += 1
            if sinais_gerados[i] == gabarito[i]:
                qtd_acertos += 1
            else:
                qtd_erros += 1
    
    acerto = (qtd_acertos - qtd_erros) / (qtd_acertos + qtd_erros) * 100
    if acerto >= ACERTO_MINIMO_TOP:
        modo_acerto = True
    elif acerto <= ACERTO_MINIMO_BOT:
        modo_acerto = False
    else:
        ocorrencias[i_grupo] = -1
        return

    acertos[i_grupo] = acerto
    ocorrencias[i_grupo] = qtd_ocorrencias

    i_grupo += 1
    if modo_acerto:
        for i_grupo in range(i_grupo, len(indices_gabarito_nao_nulo_por_grupo)):
            qtd_ocorrencias += QTD_OCORRENCIAS_NEUTRAS[i_grupo]
            indices_gabarito_nao_nulo = indices_gabarito_nao_nulo_por_grupo[i_grupo]
            i_indice = 0
            while i_indice < len(indices_gabarito_nao_nulo):
                i = indices_gabarito_nao_nulo[i_indice]
                if sinais_gerados[i]:
                    i_barras_com_ocorrencias.append(i)
                    qtd_ocorrencias += 1
                    if sinais_gerados[i] == gabarito[i]:
                        qtd_acertos += 1
                    else:
                        qtd_erros += 1
                i_indice += 1
                
            acerto = (qtd_acertos - qtd_erros) / (qtd_acertos + qtd_erros) * 100
            if acerto < ACERTO_MINIMO_TOP:
                ocorrencias[i_grupo] = -1
                return

            acertos[i_grupo] = acerto
            ocorrencias[i_grupo] = qtd_ocorrencias
    else:
        for i_grupo in range(i_grupo, len(indices_gabarito_nao_nulo_por_grupo)):
            qtd_ocorrencias += QTD_OCORRENCIAS_NEUTRAS[i_grupo]
            indices_gabarito_nao_nulo = indices_gabarito_nao_nulo_por_grupo[i_grupo]
            i_indice = 0
            while i_indice < len(indices_gabarito_nao_nulo):
                i = indices_gabarito_nao_nulo[i_indice]
                if sinais_gerados[i]:
                    i_barras_com_ocorrencias.append(i)
                    qtd_ocorrencias += 1
                    if sinais_gerados[i] == gabarito[i]:
                        qtd_acertos += 1
                    else:
                        qtd_erros += 1
                i_indice += 1
                
            acerto = (qtd_acertos - qtd_erros) / (qtd_acertos + qtd_erros) * 100
            if acerto > ACERTO_MINIMO_BOT:
                ocorrencias[i_grupo] = -1
                return

            acertos[i_grupo] = acerto
            ocorrencias[i_grupo] = qtd_ocorrencias


def calcula_rendimento_divergencias_bear(sinais_gerados_divergs_bear:np.ndarray, mem:MemCompartilhadaElos, str_param:str) -> None:
    
    # ---------------------- DIVERGENCIAS SIMPLES ----------------------
    
    # FORTE
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bear[0, 0, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bear,F,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bear,F,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # MEDIA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bear[0, 1, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bear,M,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bear,M,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            
        
    # FRACA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bear[0, 2, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bear,f,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bear,f,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            
            
    # ESCONDIDA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bear[0, 3, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bear,E,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bear,E,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    
    
    # -------------------- DIVERGENCIAS CONCAVIDADE --------------------
    
    # FORTE
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bear[1, 0, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bear,F,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bear,F,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # MEDIA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bear[1, 1, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bear,M,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bear,M,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # FRACA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bear[1, 2, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bear,f,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bear,f,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # ESCONDIDA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bear[1, 3, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bear,E,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bear,E,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            





def calcula_rendimento_divergencias_bull(sinais_gerados_divergs_bull:np.ndarray, mem:MemCompartilhadaElos, str_param:str) -> None:
    
    # ---------------------- DIVERGENCIAS SIMPLES ----------------------
    
    # FORTE
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bull[0, 0, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bull,F,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bull,F,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # MEDIA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bull[0, 1, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bull,M,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bull,M,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # FRACA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bull[0, 2, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bull,f,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bull,f,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # ESCONDIDA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bull[0, 3, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bull,E,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},S,Bull,E,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    
    
    # -------------------- DIVERGENCIAS CONCAVIDADE --------------------
    
    # FORTE
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bull[1, 0, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bull,F,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bull,F,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # MEDIA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bull[1, 1, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bull,M,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bull,M,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # FRACA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bull[1, 2, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bull,f,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bull,f,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            

    # ESCONDIDA
    i_barra_apoio = 0
    barra_apoio = 1
    for i_qtd_barras_diverg, qtd_barras_diverg in enumerate(mem.QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
        
        while i_barra_apoio < qtd_barras_diverg:
            sinais_gerados_diverg_n_barras_atras = sinais_gerados_divergs_bull[1, 3, i_barra_apoio]
        
            ocorrencias, acertos, i_barras_com_ocorrencias, eh_apenas_acerto = get_ocorrencias_divergencia(sinais_gerados_diverg_n_barras_atras,
            mem.gabarito_fechamento, mem.indices_gabarito_nao_nulo, mem.qtd_indices_gabarito_nulos_arr)
            
            if eh_apenas_acerto:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                        
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_maiores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_TOP[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bull,E,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            else:
                for i_grupo, qtd_ocorrencias in enumerate(ocorrencias):
                    if qtd_ocorrencias == -1:
                        break
                
                    diverg = Divergencia(qtd_ocorrencias)
                    if mem.ranking_menores[i_qtd_barras_diverg][i_grupo].incluir(diverg):
                        indicacao = INDICACOES_BOT[sinais_gerados_diverg_n_barras_atras[-1]]
                        diverg.armazena_representacao_em_texto(f"{str_param},C,Bull,E,{barra_apoio},{qtd_ocorrencias},{acertos[i_grupo]},{indicacao}")
                        diverg.armazena_indicacao(indicacao)
                        diverg.set_qtd_barras(barra_apoio)
                        diverg.set_i_barras_com_ocorrencias(i_barras_com_ocorrencias[:qtd_ocorrencias])
            
            i_barra_apoio += 1
            barra_apoio += 1
            



@nb.njit(cache = True)
def get_cenario_esta_bull(indicador_acima_limite_bull:np.ndarray, fonte_acima_da_sua_media:np.ndarray, fonte_extra_cresceu:np.ndarray,
    qtd_barras_solicitadas:int) -> np.ndarray:
    cenario_esta_bull = np.empty(qtd_barras_solicitadas, np.bool_)
    for i in range(-qtd_barras_solicitadas, 0):
        cenario_esta_bull[i] = fonte_acima_da_sua_media[i] and indicador_acima_limite_bull[i] and fonte_extra_cresceu[i]
    return cenario_esta_bull

@nb.njit(cache = True)
def get_cenario_esta_bear(indicador_abaixo_limite_bear:np.ndarray, fonte_abaixo_da_sua_media:np.ndarray, fonte_extra_decresceu:np.ndarray,
    qtd_barras_solicitadas:int) -> np.ndarray:
    cenario_esta_bear = np.empty(qtd_barras_solicitadas, np.bool_)
    for i in range(-qtd_barras_solicitadas, 0):
        cenario_esta_bear[i] = fonte_abaixo_da_sua_media[i] and indicador_abaixo_limite_bear[i] and fonte_extra_decresceu[i]
    return cenario_esta_bear


@nb.njit(cache = True)
def get_fonte_divergencia_cresceu_e_formou_pico(fonte_divergencia_valor:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Verifica varias barras da fonte recebida. Verifica onde a fonte cresceu e onde ela formou um pico.
    Formar pico em d0 significa que em 1d e d1 a fonte era mais alta.
    """
    
    fonte_cresceu = np.empty(len(fonte_divergencia_valor), np.bool_)
    formou_pico = np.empty(len(fonte_divergencia_valor), np.bool_)
    
    fonte_cresceu[0] = False
    formou_pico[0] = False
    
    i = 1
    while i < len(fonte_divergencia_valor) - 1:
        if fonte_divergencia_valor[i] >= fonte_divergencia_valor[i - 1]:
            fonte_cresceu[i] = True
            # Talvez formou pico, verifica a barra seguinte
            if fonte_divergencia_valor[i + 1] < fonte_divergencia_valor[i]:
                formou_pico[i] = True
                i += 1
                fonte_cresceu[i] = False
                formou_pico[i] = False
            elif fonte_divergencia_valor[i + 1] == fonte_divergencia_valor[i]:
                formou_pico[i] = True
            else:
                formou_pico[i] = False
                
        else:
            fonte_cresceu[i] = False
            formou_pico[i] = False
        i += 1
    
    i = len(fonte_divergencia_valor) - 1
    fonte_cresceu[i] = fonte_divergencia_valor[i] >= fonte_divergencia_valor[i - 1]
    formou_pico[i] = False
    
    return fonte_cresceu, formou_pico

@cache
def get_fonte_divergencia_cresceu_e_formou_pico_cache(fonte_divergencia:Fonte) -> tuple[np.ndarray, np.ndarray]:
    return get_fonte_divergencia_cresceu_e_formou_pico(fonte_divergencia.valor)
 
##################################################################
##################################################################

@nb.njit(cache = True)
def get_divergs_bear_indicador_vs_fonte(fonte_divergencia:np.ndarray, 
                    indicador:np.ndarray, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA:int,
                    QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS:int) -> np.ndarray:
    """Para cada barra de backtest, avalia que divergencias ocorreu no dia. É retornado um
    array para cada tipo de divergencia: forte, media, fraca e escondida.
    Cada posição do array é referente a uma quantidade de dias atrás e a outra posição
    referente a barra no gráfico
    
    Ex:
            [
                [ Forte
                    [0, 0, 1, 1, ...], Divergencia 1d
                    [1, 0, 0, 1, ...], Divergencia 2d
                    [1, 1, 0, 1, ...], Divergencia 3d
                    ...
                ],
                [ Media
                    [1, 1, 1, 1, ...], Divergencia 1d
                    [0, 0, 1, 1, ...], Divergencia 2d
                    [0, 0, 0, 1, ...], Divergencia 3d
                    ...
                ],
                [ Fraca
                    [1, 1, 1, 1, ...], Divergencia 1d
                    [0, 0, 0, 1, ...], Divergencia 2d
                    [0, 0, 1, 1, ...], Divergencia 3d
                    ...
                ],
                [ Escondida
                    [0, 0, 1, 1, ...], Divergencia 1d
                    [1, 0, 0, 1, ...], Divergencia 2d
                    [1, 1, 0, 1, ...], Divergencia 3d
                    ...
                
                ]
            ]
            > Dia 0: Sem divergencia forte.
            > Dia 1: Sem divergencia forte.
            > Dia 2: Divergencia com as barras 1d, 2d, 3d.

    Args:
        fonte_da_divergencia (np.ndarray): Fonte qualquer.
        indicador (np.ndarray): Indicador avaliado.
        QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA (int): Numero de barras avaliadas, contando a barra
        do dia seguinte.
        QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS (int): Numero de barras que se busca divergencia entre o
        indicador e fonte.

    Returns:
        np.ndarray
    """
    
    divergencias_bear = np.empty((4, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA), np.int8)
    
    for i_barra_hoje in range(-QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, 0):
        
        # Busca divergencia pra d1 em diante
        # "dia" indica a quantos dias ocorreu a divergencia
        for i_barra_compara, i_barra in enumerate(range(i_barra_hoje - 1, i_barra_hoje - QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS - 1, -1)):
            divergencias_bear[0, i_barra_compara, i_barra_hoje] = fez_diverg_bear_forte(fonte_divergencia,
            indicador, i_barra_hoje, i_barra)

            divergencias_bear[1, i_barra_compara, i_barra_hoje] = fez_diverg_bear_media(fonte_divergencia,
            indicador, i_barra_hoje, i_barra)

            divergencias_bear[2, i_barra_compara, i_barra_hoje] = fez_diverg_bear_fraca(fonte_divergencia,
            indicador, i_barra_hoje, i_barra)

            divergencias_bear[3, i_barra_compara, i_barra_hoje] = fez_diverg_bear_escondida(fonte_divergencia,
            indicador, i_barra_hoje, i_barra)
        
    return divergencias_bear


######
@cache
def get_divergs_bear_simples_e_concav(fonte_divergencia:Fonte, indicador:Indicador, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA:int,
                              QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS:int) -> np.ndarray:
    """É comparado uma fonte e um indicador e verificada a divergencia entre os dois. As divergencia são da barra[0]
    até a barra[1], depois até a barra[2], até chegar a barra[QTD_BARRAS_PARA_BURSCAR_DIVERGS].
    Retorna um array que informa com "1" onde ocorreu divergencia e "0" onde nao ocorreu.

    Args:
        fonte_divergencia (Fonte): Fonte qualquer que está sendo comparada com o indicador.
        indicador (Indicador): Indicador que será comparado a fonte.
        QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA (np.ndarray): Numero de barras graficas avaliadas.
        QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS (int): Número de barras atrás para buscar a divergencia.

    Returns:
        np.ndarray: Array que informa todas as informações sobre divergencia, ele possui 4 dimensões:
        1 - Simples   Concavidade
        2 - Forte   Media   Fraca   Concavidade
        3 - Divergencia 1d, 2d, 3d, ..., QTD_BARRASd
        4 - Barra do gráfico.
    """
    
    fonte_cresceu, formou_pico = get_fonte_divergencia_cresceu_e_formou_pico_cache(fonte_divergencia)
    # Cenario -> Cresceu/Pico -> Gráfica
    divergs_indicador_vs_fonte =  get_divergs_bear_indicador_vs_fonte(fonte_divergencia.valor,
            indicador.valor, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS)
    divergs_bear_simples_e_concav = np.empty((2, 4, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA), np.int8)
    
    for i_barra in range(-QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, 0):
        
        dia = 1
        i_dia = 0
        if fonte_cresceu[i_barra - dia]:
            divergs_bear_simples_e_concav[0, 0, i_dia, i_barra] = divergs_indicador_vs_fonte[0, i_dia, i_barra]
            divergs_bear_simples_e_concav[1, 0, i_dia, i_barra] = divergs_indicador_vs_fonte[0, i_dia, i_barra]
            
            divergs_bear_simples_e_concav[0, 1, i_dia, i_barra] = divergs_indicador_vs_fonte[1, i_dia, i_barra]
            divergs_bear_simples_e_concav[1, 1, i_dia, i_barra] = divergs_indicador_vs_fonte[1, i_dia, i_barra]
            
            divergs_bear_simples_e_concav[0, 2, i_dia, i_barra] = divergs_indicador_vs_fonte[2, i_dia, i_barra]
            divergs_bear_simples_e_concav[1, 2, i_dia, i_barra] = divergs_indicador_vs_fonte[2, i_dia, i_barra]

            divergs_bear_simples_e_concav[0, 3, i_dia, i_barra] = divergs_indicador_vs_fonte[3, i_dia, i_barra]
            divergs_bear_simples_e_concav[1, 3, i_dia, i_barra] = divergs_indicador_vs_fonte[3, i_dia, i_barra]
        else:
            divergs_bear_simples_e_concav[0, 0, i_dia, i_barra] = 0
            divergs_bear_simples_e_concav[1, 0, i_dia, i_barra] = 0
            
            divergs_bear_simples_e_concav[0, 1, i_dia, i_barra] = 0
            divergs_bear_simples_e_concav[1, 1, i_dia, i_barra] = 0
            
            divergs_bear_simples_e_concav[0, 2, i_dia, i_barra] = 0
            divergs_bear_simples_e_concav[1, 2, i_dia, i_barra] = 0

            divergs_bear_simples_e_concav[0, 3, i_dia, i_barra] = 0
            divergs_bear_simples_e_concav[1, 3, i_dia, i_barra] = 0
        
        for dia in range(2, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS + 1):
            i_dia = dia - 1
            if fonte_cresceu[i_barra - dia]:
                divergs_bear_simples_e_concav[0, 0, i_dia, i_barra] = divergs_indicador_vs_fonte[0, i_dia, i_barra]
                divergs_bear_simples_e_concav[0, 1, i_dia, i_barra] = divergs_indicador_vs_fonte[1, i_dia, i_barra]
                divergs_bear_simples_e_concav[0, 2, i_dia, i_barra] = divergs_indicador_vs_fonte[2, i_dia, i_barra]
                divergs_bear_simples_e_concav[0, 3, i_dia, i_barra] = divergs_indicador_vs_fonte[3, i_dia, i_barra]
            else:
                divergs_bear_simples_e_concav[0, 0, i_dia, i_barra] = 0
                divergs_bear_simples_e_concav[0, 1, i_dia, i_barra] = 0
                divergs_bear_simples_e_concav[0, 2, i_dia, i_barra] = 0
                divergs_bear_simples_e_concav[0, 3, i_dia, i_barra] = 0
            
            if formou_pico[i_barra - dia]:
                divergs_bear_simples_e_concav[1, 0, i_dia, i_barra] = divergs_indicador_vs_fonte[0, i_dia, i_barra]
                divergs_bear_simples_e_concav[1, 1, i_dia, i_barra] = divergs_indicador_vs_fonte[1, i_dia, i_barra]
                divergs_bear_simples_e_concav[1, 2, i_dia, i_barra] = divergs_indicador_vs_fonte[2, i_dia, i_barra]
                divergs_bear_simples_e_concav[1, 3, i_dia, i_barra] = divergs_indicador_vs_fonte[3, i_dia, i_barra]
            else:
                divergs_bear_simples_e_concav[1, 0, i_dia, i_barra] = 0
                divergs_bear_simples_e_concav[1, 1, i_dia, i_barra] = 0
                divergs_bear_simples_e_concav[1, 2, i_dia, i_barra] = 0
                divergs_bear_simples_e_concav[1, 3, i_dia, i_barra] = 0

    return divergs_bear_simples_e_concav


def get_sinais_gerados_divergs_bear(fonte_divergencia:Fonte, indicador:Indicador,
                                    cenario_esta_bull:np.ndarray, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS:int) -> np.ndarray:
    """É comparado uma fonte e um indicador e verificada a divergencia entre os dois. As divergencia são da barra[0]
    até a barra[1], depois até a barra[2], até chegar a barra[QTD_BARRAS_PARA_BURSCAR_DIVERGS].
Para gerar os sinais, precisa do cenário estar bull, então verifica ele também.

    Args:
        fonte_divergencia (Fonte): Fonte qualquer que está sendo comparada com o indicador.
        indicador (Indicador): Indicador que será comparado a fonte.
        cenario_esta_bull (np.ndarray): Para cada barra, informa se o cenário é bull ou não.
        QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS (int): Número de barras atrás para buscar a divergencia.

    Returns:
        np.ndarray: Array que informa todas as informações sobre divergencia, ele possui 4 dimensões:
        1 - Simples   Concavidade
        2 - Forte   Media   Fraca   Concavidade
        3 - Divergencia 1d, 2d, 3d, ..., QTD_BARRASd
        4 - Barra do gráfico.
    """
    divergs_bear_simples_e_concav = get_divergs_bear_simples_e_concav(
        fonte_divergencia, indicador, len(cenario_esta_bull),
        QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS)
    
    return np.where(cenario_esta_bull, divergs_bear_simples_e_concav, 0)

def main_divergencias_bear(mem:MemCompartilhadaElos) -> None:
    
    cenario_esta_bull = get_cenario_esta_bull(mem.indicador_acima_limite_bull, mem.fonte_acima_da_sua_media,
                    mem.fonte_extra_cresceu, mem.QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA)
        
    fontes_divergencia : list[Fonte] = mem.fontes_divergencia
    indicador : Indicador = mem.indicador
    QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS : int = mem.QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS
    
    for fonte_divergencia in fontes_divergencia:
        
        sinais_gerados_divergs_bear = get_sinais_gerados_divergs_bear(fonte_divergencia, indicador,
                                    cenario_esta_bull, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS)

        str_param = f"{mem.STR_ELO_MACRO_SUAVIZACAO},{mem.STR_ELO_MACRO_FONTE_EXTRA},{mem.STR_ELO_MACRO_INDICADOR},{fonte_divergencia}"        
        calcula_rendimento_divergencias_bear(sinais_gerados_divergs_bear, mem, str_param)


@nb.njit(cache = True)
def get_fonte_divergencia_decresceu_e_formou_vale(fonte_divergencia_valor:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Verifica varias barras da fonte recebida. Verifica onde a fonte decresceu e onde ela formou um vale.
    Formar vale em d0 significa que em 1d e d1 a fonte era mais alta.
    """
    
    fonte_decresceu = np.empty(len(fonte_divergencia_valor), np.bool_)
    formou_vale = np.empty(len(fonte_divergencia_valor), np.bool_)
    
    fonte_decresceu[0] = False
    formou_vale[0] = False
    
    i = 1
    while i < len(fonte_divergencia_valor) - 1:
        if fonte_divergencia_valor[i] <= fonte_divergencia_valor[i - 1]:
            fonte_decresceu[i] = True
            # Talvez formou pico, verifica a barra seguinte
            if fonte_divergencia_valor[i + 1] > fonte_divergencia_valor[i]:
                formou_vale[i] = True
                i += 1
                fonte_decresceu[i] = False
                formou_vale[i] = False
            elif fonte_divergencia_valor[i + 1] == fonte_divergencia_valor[i]:
                formou_vale[i] = True
            else:
                formou_vale[i] = False
                
        else:
            fonte_decresceu[i] = False
            formou_vale[i] = False
        i += 1
    
    i = len(fonte_divergencia_valor) - 1
    fonte_decresceu[i] = fonte_divergencia_valor[i] <= fonte_divergencia_valor[i - 1]
    formou_vale[i] = False
    
    return fonte_decresceu, formou_vale

@cache
def get_fonte_divergencia_decresceu_e_formou_vale_cache(fonte_divergencia:Fonte) -> tuple[np.ndarray, np.ndarray]:
    return get_fonte_divergencia_decresceu_e_formou_vale(fonte_divergencia.valor)

@nb.njit(cache = True)
def get_divergs_bull_indicador_vs_fonte(fonte_divergencia:np.ndarray, 
                    indicador:np.ndarray, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA:int,
                    QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS:int) -> np.ndarray:
    """Para cada barra de backtest, avalia que divergencias ocorreu no dia. É retornado um
    array para cada tipo de divergencia: forte, media, fraca e escondida.
    Cada posição do array é referente a uma quantidade de dias atráse a outra posição
    referente a barra no gráfico
    
    Ex:
            [
            [ Forte
                [0, 0, 1, 1, ...], Divergencia 1d
                [1, 0, 0, 1, ...], Divergencia 2d
                [1, 1, 0, 1, ...], Divergencia 3d
                ...
            ]
            [ Media
                [1, 1, 1, 1, ...], Divergencia 1d
                [0, 0, 1, 1, ...], Divergencia 2d
                [0, 0, 0, 1, ...], Divergencia 3d
                ...
            ]
            [ Fraca
                [1, 1, 1, 1, ...], Divergencia 1d
                [0, 0, 0, 1, ...], Divergencia 2d
                [0, 0, 1, 1, ...], Divergencia 3d
                ...
            ]
            [ Escondida
                [0, 0, 1, 1, ...], Divergencia 1d
                [1, 0, 0, 1, ...], Divergencia 2d
                [1, 1, 0, 1, ...], Divergencia 3d
                ...
            
            ]
            > Dia 0: Sem divergencia forte.
            > Dia 1: Sem divergencia forte.
            > Dia 2: Divergencia com as barras 1d, 2d, 3d.

    Args:
        fonte_da_divergencia (np.ndarray): Fonte qualquer.
        indicador (np.ndarray): Indicador avaliado.
        QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA (int): Numero de barras avaliadas, contando a barra
        do dia seguinte.
        QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS (int): Numero de barras que se busca divergencia entre o
        indicador e fonte.

    Returns:
        np.ndarray
    """
    
    divergencias_bull = np.empty((4, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA), np.int8)
    
    for i_barra_hoje in range(-QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, 0):
        
        # Busca divergencia pra d1 em diante
        # "dia" indica a quantos dias ocorreu a divergencia
        for i_barra_compara, i_barra in enumerate(range(i_barra_hoje - 1, i_barra_hoje - QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS - 1, -1)):
            divergencias_bull[0, i_barra_compara, i_barra_hoje] = fez_diverg_bull_forte(fonte_divergencia,
            indicador, i_barra_hoje, i_barra)

            divergencias_bull[1, i_barra_compara, i_barra_hoje] = fez_diverg_bull_media(fonte_divergencia,
            indicador, i_barra_hoje, i_barra)

            divergencias_bull[2, i_barra_compara, i_barra_hoje] = fez_diverg_bull_fraca(fonte_divergencia,
            indicador, i_barra_hoje, i_barra)

            divergencias_bull[3, i_barra_compara, i_barra_hoje] = fez_diverg_bull_escondida(fonte_divergencia,
            indicador, i_barra_hoje, i_barra)
        
    return divergencias_bull

######
@cache
def get_divergs_bull_simples_e_concav(fonte_divergencia:Fonte, indicador:Indicador, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA:int,
                              QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS:int) -> np.ndarray:
    """É comparado uma fonte e um indicador e verificada a divergencia entre os dois. As divergencia são da barra[0]
    até a barra[1], depois até a barra[2], até chegar a barra[QTD_BARRAS_PARA_BURSCAR_DIVERGS].
    Retorna um array que informa com "1" onde ocorreu divergencia e "0" onde nao ocorreu.

    Args:
        fonte_divergencia (Fonte): Fonte qualquer que está sendo comparada com o indicador.
        indicador (Indicador): Indicador que será comparado a fonte.
        QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA (np.ndarray): Numero de barras graficas avaliadas.
        QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS (int): Número de barras atrás para buscar a divergencia.

    Returns:
        np.ndarray: Array que informa todas as informações sobre divergencia, ele possui 4 dimensões:
        1 - Simples   Concavidade
        2 - Forte   Media   Fraca   Concavidade
        3 - Divergencia 1d, 2d, 3d, ..., QTD_BARRASd
        4 - Barra do gráfico.
    """
    
    fonte_decresceu, formou_vale = get_fonte_divergencia_decresceu_e_formou_vale_cache(fonte_divergencia)
    # Cenario -> Decresceu/Vale -> Gráfica
    divergs_indicador_vs_fonte =  get_divergs_bull_indicador_vs_fonte(fonte_divergencia.valor,
            indicador.valor, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS)
    divergs_bull_simples_e_concav = np.empty((2, 4, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA), np.int8)
    
    for i_barra in range(-QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, 0):
        
        dia = 1
        i_dia = 0
        if fonte_decresceu[i_barra - dia]:
            divergs_bull_simples_e_concav[0, 0, i_dia, i_barra] = divergs_indicador_vs_fonte[0, i_dia, i_barra]
            divergs_bull_simples_e_concav[1, 0, i_dia, i_barra] = divergs_indicador_vs_fonte[0, i_dia, i_barra]
            
            divergs_bull_simples_e_concav[0, 1, i_dia, i_barra] = divergs_indicador_vs_fonte[1, i_dia, i_barra]
            divergs_bull_simples_e_concav[1, 1, i_dia, i_barra] = divergs_indicador_vs_fonte[1, i_dia, i_barra]
            
            divergs_bull_simples_e_concav[0, 2, i_dia, i_barra] = divergs_indicador_vs_fonte[2, i_dia, i_barra]
            divergs_bull_simples_e_concav[1, 2, i_dia, i_barra] = divergs_indicador_vs_fonte[2, i_dia, i_barra]

            divergs_bull_simples_e_concav[0, 3, i_dia, i_barra] = divergs_indicador_vs_fonte[3, i_dia, i_barra]
            divergs_bull_simples_e_concav[1, 3, i_dia, i_barra] = divergs_indicador_vs_fonte[3, i_dia, i_barra]
        else:
            divergs_bull_simples_e_concav[0, 0, i_dia, i_barra] = 0
            divergs_bull_simples_e_concav[1, 0, i_dia, i_barra] = 0
            
            divergs_bull_simples_e_concav[0, 1, i_dia, i_barra] = 0
            divergs_bull_simples_e_concav[1, 1, i_dia, i_barra] = 0
            
            divergs_bull_simples_e_concav[0, 2, i_dia, i_barra] = 0
            divergs_bull_simples_e_concav[1, 2, i_dia, i_barra] = 0

            divergs_bull_simples_e_concav[0, 3, i_dia, i_barra] = 0
            divergs_bull_simples_e_concav[1, 3, i_dia, i_barra] = 0
        
        for dia in range(2, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS + 1):
            i_dia = dia - 1
            if fonte_decresceu[i_barra - dia]:
                divergs_bull_simples_e_concav[0, 0, i_dia, i_barra] = divergs_indicador_vs_fonte[0, i_dia, i_barra]
                divergs_bull_simples_e_concav[0, 1, i_dia, i_barra] = divergs_indicador_vs_fonte[1, i_dia, i_barra]
                divergs_bull_simples_e_concav[0, 2, i_dia, i_barra] = divergs_indicador_vs_fonte[2, i_dia, i_barra]
                divergs_bull_simples_e_concav[0, 3, i_dia, i_barra] = divergs_indicador_vs_fonte[3, i_dia, i_barra]
            else:
                divergs_bull_simples_e_concav[0, 0, i_dia, i_barra] = 0
                divergs_bull_simples_e_concav[0, 1, i_dia, i_barra] = 0
                divergs_bull_simples_e_concav[0, 2, i_dia, i_barra] = 0
                divergs_bull_simples_e_concav[0, 3, i_dia, i_barra] = 0
            
            if formou_vale[i_barra - dia]:
                divergs_bull_simples_e_concav[1, 0, i_dia, i_barra] = divergs_indicador_vs_fonte[0, i_dia, i_barra]
                divergs_bull_simples_e_concav[1, 1, i_dia, i_barra] = divergs_indicador_vs_fonte[1, i_dia, i_barra]
                divergs_bull_simples_e_concav[1, 2, i_dia, i_barra] = divergs_indicador_vs_fonte[2, i_dia, i_barra]
                divergs_bull_simples_e_concav[1, 3, i_dia, i_barra] = divergs_indicador_vs_fonte[3, i_dia, i_barra]
            else:
                divergs_bull_simples_e_concav[1, 0, i_dia, i_barra] = 0
                divergs_bull_simples_e_concav[1, 1, i_dia, i_barra] = 0
                divergs_bull_simples_e_concav[1, 2, i_dia, i_barra] = 0
                divergs_bull_simples_e_concav[1, 3, i_dia, i_barra] = 0

    return divergs_bull_simples_e_concav


def get_sinais_gerados_divergs_bull(fonte_divergencia:Fonte, indicador:Indicador,
                                    cenario_esta_bear:np.ndarray, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS:int) -> np.ndarray:
    """É comparado uma fonte e um indicador e verificada a divergencia entre os dois. As divergencia são da barra[0]
    até a barra[1], depois até a barra[2], até chegar a barra[QTD_BARRAS_PARA_BURSCAR_DIVERGS].
Para gerar os sinais, precisa do cenário estar bear, então verifica ele também.

    Args:
        fonte_divergencia (Fonte): Fonte qualquer que está sendo comparada com o indicador.
        indicador (Indicador): Indicador que será comparado a fonte.
        cenario_esta_bear (np.ndarray): Para cada barra, informa se o cenário é bear ou não.
        QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS (int): Número de barras atrás para buscar a divergencia.

    Returns:
        np.ndarray: Array que informa todas as informações sobre divergencia, ele possui 4 dimensões:
        1 - Simples   Concavidade
        2 - Forte   Media   Fraca   Concavidade
        3 - Divergencia 1d, 2d, 3d, ..., QTD_BARRASd
        4 - Barra do gráfico.
    """
    divergs_bull_simples_e_concav = get_divergs_bull_simples_e_concav(
        fonte_divergencia, indicador, len(cenario_esta_bear),
        QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS)
    
    return np.where(cenario_esta_bear, divergs_bull_simples_e_concav, 0)
    
def main_divergencias_bull(mem:MemCompartilhadaElos) -> None:
    
    cenario_esta_bear = get_cenario_esta_bear(mem.indicador_abaixo_limite_bear, mem.fonte_abaixo_da_sua_media,
                    mem.fonte_extra_decresceu, mem.QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA)
        
    fontes_divergencia : list[Fonte] = mem.fontes_divergencia
    indicador : Indicador = mem.indicador
    QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS : int = mem.QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS
    
    for fonte_divergencia in fontes_divergencia:
        
        sinais_gerados_divergs_bull = get_sinais_gerados_divergs_bull(fonte_divergencia, indicador,
                                    cenario_esta_bear, QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS)

        str_param = f"{mem.STR_ELO_MACRO_SUAVIZACAO},{mem.STR_ELO_MACRO_FONTE_EXTRA},{mem.STR_ELO_MACRO_INDICADOR},{fonte_divergencia}"
        calcula_rendimento_divergencias_bull(sinais_gerados_divergs_bull, mem, str_param)

def limpa_cache_por_indicador():
    get_divergs_bear_simples_e_concav.cache_clear()
    get_divergs_bull_simples_e_concav.cache_clear()


@cache
def get_espacos_em_branco_indicador(colunas_ocupadas_indicador:int) -> str:
    """Recebe o o numero de colunas que o indicador escolhido ocupa na planilha. Retorna
    o numero de espaços em branco que devem ser inseridos após ele.
    Para fazer a conta, verifica todos os indicadores escolhidos. O maior dos indicadores
    não possui espaços em branco, pois já ocupa a extensão máxima da planilha. Os devem
    ocupar a diferença como espaços em branco.

    Args:
        colunas_ocupadas_indicador (int): Numero de colunas que o indicador ocupa na planilha.

    Returns:
        str: Espaçõs em branco inseridos após o indicador. É representado por "#"
    """
    qtd_colunas_extras = COLUNAS_USADAS_POR_INDICADOR_ARQUIVO_SAIDA - colunas_ocupadas_indicador
    return ",#"*qtd_colunas_extras


def get_indice_superposicao(valores_para_avaliar:list[int], todos_valores:list[int]
) -> tuple[float, dict[int, int]]:
    """Recebe duas listas. A primeira com os valores para se incluir na segunda e a segunda
    com todos os valores já inclusos. Determina o % de valores da primeira que estão na segunda
    lista (indice de superposição). Também, é calculado um dicionário que armazena os elementos
    que podem ser incluidos sem causar redundancia a lista.

    Args:
        valores_para_avaliar (list[int]): Valores para serem incluidos.
        todos_valores (list[int]): Valores já incluidos.

    Returns:
        tuple[float, dict[int, int]]: 
        superposicao (float)
        indices_incluir (dict[int, int]): Indice de todos_valores : Valor para incluir
    """
    i_inicio_busca = 0
    i_fim_busca = len(todos_valores) - 1
    # Guarda os valores avaliados que nao tem copia em "todos_valores"
    valores_para_incluir = []
    # Guarda os indices de "todos_valores" para incluir os objetos acima
    indices_onde_incluir = []
    
    def busca_bin(valor_para_buscar:int, inicio:int, fim:int) -> int:
        """Faz busca binaria na lista "todos_valores" do valor "valor_para_buscar". Há dois cenários
        possíveis:
        
        - Encontrou
        Retorna seu indice e NÃO inlcui no valores_para_incluir, pois já existe uma cópia do elemento.
        
        - Não encontrou
        Retorna -1 e INCLUI no valores_para_incluir.

        Args:
            valor_para_buscar (int): Elemento a ser buscado.
            inicio (int): Indice de inicio da busca.
            fim (int): Indice de fim da busca. 

        Returns:
            int: -1 caso não encontrou, Indice caso encontrou.
        """
        if fim > inicio:
            meio = (inicio + fim)//2
            if valor_para_buscar > todos_valores[meio]:
                return busca_bin(valor_para_buscar, inicio, meio - 1)
            elif valor_para_buscar < todos_valores[meio]:
                return busca_bin(valor_para_buscar, meio + 1, fim)
            else:
                return meio
        
        elif inicio == fim:
            if valor_para_buscar > todos_valores[inicio]:
                valores_para_incluir.append(inicio)
                return -1
            elif valor_para_buscar < todos_valores[inicio]:
                valores_para_incluir.append(inicio + 1)
                return -1
            else:
                return inicio
        
        else:
            valores_para_incluir.append(inicio)
            return -1

    
    qtd_superposicoes = 0
    for valor_para_buscar in valores_para_avaliar:
        indice_match = busca_bin(valor_para_buscar, i_inicio_busca, i_fim_busca)
        if indice_match != -1:
        # Encontrou o valor. Conta uma superposição e move o indice de inicio da busca adiante
            qtd_superposicoes += 1
            i_inicio_busca = indice_match + 1
        else:
        # Nao houve match, entao o indice pode ser incluido sem causar repeticoes
            indices_onde_incluir.append(valor_para_buscar)
    
    # Inverter os vetores, pois eles estao em ordem descrescente. Inserção em ordem crescente dá problema
    indices_incluir = {
        indice:valor_incluir for indice, valor_incluir in zip(indices_onde_incluir[::-1], valores_para_incluir[::-1])
    }
    return qtd_superposicoes/len(valores_para_avaliar)*100, indices_incluir


def amostra_esta_no_conjunto_por_superposicao(valores_para_avaliar:list[int], todos_valores:list[int], superposicao_aceitavel:float) -> bool:
    """Verifica se uma amostra (valores_para_avaliar) já está presente em um conjunto (todos_valores) atraves
    do % de elementos da amostra que se encontram no conjunto (superposição).
    Caso a amostra esteja presente, não faz a inclusão dela, do contrário faz sua inclusão.
    Usada impedir a existencia de repetições de elementos, quando os elementos podem ser compardos por listas.

    Args:
        valores_para_avaliar (list[int]): Amostra de valores.
        todos_valores (list[int]): Conjunto com todas as amostras já incluídas.
        superposicao_aceitavel (float): Percentual de elementos da amostra que podem estar no
        conjunto. Se estiver acima do aceitavel, então a amostra é considerada já inclusa no
        conjunto e não é feita inclusão.

    Returns:
        bool: Se a amostra foi incluída ou não.
    """
    assert 0 <= superposicao_aceitavel < 100, "Superposicao deve ser entre 0 e 100 (exclusivo)."

    superposicao, indices_incluir = get_indice_superposicao(valores_para_avaliar, todos_valores)
    if superposicao <= superposicao_aceitavel:
    
        for indice, valor_incluir in indices_incluir.items():
            todos_valores.insert(valor_incluir, indice)
        return True

    return False