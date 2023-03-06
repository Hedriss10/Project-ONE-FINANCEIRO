from __future__ import annotations
from math import floor, sqrt
import numpy as np
from numpy import nan
import numba as nb
from itertools import combinations_with_replacement, product
from talib import LINEARREG as linreg
from talib import DEMA  as dema
from talib import TEMA  as tema
from talib import KAMA  as kama
from talib import MAMA  as mama
from functools import lru_cache

nomes_medias = {
0  : "EMA",     # E    Media Exponencial
1  : "SMA",     # A    Media Aritmetica
2  : "WMA",     # P    Media Ponderada
3  : "HMA",     # H    Media de Hull
4  : "SMMA",    # S    Media Suavizada
5  : "ALMA",    # AL   Media de Arnaud Legoux
6  : "LINREG",  # MQ   Media dos Minimos Quadrados
7  : "DEMA",    # ED   Media Exponencial Dupla
8  : "TEMA",    # ET   Media Exponencial Tripla
9  : "VWMA",    # VP   Media do Volume Ponderada
10 : "KAMA",    # K    Media Movel Adaptativa De Kaufman (KAMA)
11 : "MAMA",    # MA   Media Movel Adaptativa MESA (MAMA)
12 : "FAMA",    # FA   Following MAMA (FAMA)
13 : "TRIMA",   # T    Media Movel Triangular
14 : "SMM",     # ME   Mediana Movel
15 : "VAMA",   # VA   Media Movel Ajustada ao Volume
16 : "ZLEMA",   # ZL   Media Movel Exponencial Zero Lag
17 : "EVWMA",    # EV   Media Movel Elastica do Volume
18 : "FRAMA",   # FR   Media Movel Adaptavel Fractal
19 : "DSMA",    # DP   Media Movel Escalada do Desvio Padrão (DSMA)
20 : "VIDYA",   # IV   Media Movel de Índice Variável (VIDYA)
21 : "FRAMA2",  # FR2  Sem limites no alpha
22 : "DSMA2",   # DP2  Com limites no alpha
23 : "EWMA",    # PE   Media Ponderada Exponencialmente
24 : "ICHI_A",  #      Ichimoku (senkou_A)     
24 : "ICHI_B",  #      Ichimoku (senkou_B)     
24 : "ICHI_T",  #      Ichimoku (tenkan_sen)     
}


dict_nome_media_cod = {sigla:cod for cod, sigla in nomes_medias.items()}

def get_nome_media(codigo:int) -> str:
    return nomes_medias[codigo]

def get_codigo_media(nome:str) -> int:
    return dict_nome_media_cod[nome]

@lru_cache(1)
def qtd_medias() -> int:
    return len(nomes_medias)

def combina_medias(periodos, tipos):
    """ A partir dos iteraveis peridos e tipos, retorna as combinações de duas médias.
    A iteração deve ser feita por:
    for periodos, tipos in retorno:
        periodos[0], periodos[1], tipos[0], tipos[1]"""
    
    # Atualmente fora de uso, permite apenas combinacoes de mesmo perido
    #periodos_combinados = list(combinations_with_replacement(periodos, 2))
    periodos_combinados = [(i, i) for i in periodos]
    #tipos_combinados = list(combinations_with_replacement(tipos, 2))
    #tipos_cartesiano = list(product(tipos, repeat = 2))
    tipos_combinados = [(i, i) for i in tipos]
    tipos_cartesiano = [(i, i) for i in tipos]
	
    temp = [
	list(product([(p1,p2)], tipos_combinados, repeat = 1)) if
	p1 == p2 else
	list(product([(p1,p2)], tipos_cartesiano, repeat = 1)) for p1, p2 in periodos_combinados]

    l = []
    for x1 in temp:
        for x2 in x1:
            l.append(np.array((np.array(x2[0]), np.array(x2[1]))))
    return np.array(l)


# Combinacoes de arr1 com arr2. Nao permite combinacoes repetidas
# do tipo (a, b) (b, a)
def combs(arr1, arr2):
	combinacoes = []
	indices_j = list(range(len(arr2)))
	for i in range(len(arr1)):
		intrsc = []
		for j in indices_j:
			print(indices_j, (i, j))
			combinacoes.append((arr1[i],arr2[j]))
			if arr1[i] == arr2[j]:
				intrsc.append(j)
		for j in intrsc:
			if set(arr1[i + 1:]).intersection(set(arr2[j + 1:])):
				indices_j.remove(j)
	return combinacoes


class CombinacoesMedias:
    def __init__(self, parametros:dict, grau_liberdade_quarta_media = None):
        """
        Lida com a combinatória de duas médias com mesmo conjunto de parâmetros.
        Ou seja, mesmos fontes, periodos, etc. atraves da funcao combinacoes\n
        O grau de liberdade pode ser 3 opcoes:
        1 - Médias idênticas
        2 - Médias de mesmo tipo
        3 - Médias independentes"""

        self.tipos_ma_1 = np.array(parametros["Tipos"])
        fontes_1        = np.array(parametros["Fontes"])
        periodos_1      = np.array(parametros["Periodos"])
        self.periodos_1 = periodos_1
        offsets_1       = np.array(parametros["Offsets"])
        sigmas_1        = np.array(parametros["Sigmas"])
        lims_rapido_1   = np.array(parametros["Lims Rapido"])
        self.lims_rapido_1 = lims_rapido_1
        lims_lento_1    = np.array(parametros["Lims Lento"])
        self.lims_lento_1 = lims_lento_1

        if grau_liberdade_quarta_media == None: # Nao importa quando eh None, apenas esta aqui para inicializar
            self.combinacoes =  self.combinacoes_independentes
        elif grau_liberdade_quarta_media == 1:
            self.combinacoes =  self.combinacoes_identicas
        elif grau_liberdade_quarta_media == 2: # NÃO PEGA PRODUTO CARTESIANO ENTRE MAMA/FAMA
            self.combinacoes =  self.combinacoes_mesmo_tipo
        elif grau_liberdade_quarta_media == 3:
            self.combinacoes =  self.combinacoes_independentes

        # Funcao que combina os tipos de media
        if grau_liberdade_quarta_media == 1 or grau_liberdade_quarta_media == 2:
            self.combinacoes_tipos = self.combinacoes_tipos_identicos
        else:
            self.combinacoes_tipos = self.combinacoes_tipos_independentes

        def combina_media(fontes, periodos, offsets, sigmas, lims_rapido, lims_lento):
            return np.array(list(product(fontes, periodos, offsets, sigmas, lims_rapido, lims_lento)), dtype = np.object)
            
        vet_vazio_int = np.zeros(1, np.int64)
        vetor_vazio = np.zeros(1)
        # Combinacao de media em funcao do tipo escolhido
        self.comb_por_media_1 = {
        #-----------------------------------------------------------------------------------------------------
        # Tipo                  Fonte     Periodo          Offset       Sigma     Limite Rapido   Limite Lento
        #-----------------------------------------------------------------------------------------------------
            0  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            1  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            2  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            3  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            4  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            5  : combina_media(fontes_1, periodos_1,     offsets_1,   sigmas_1,    vetor_vazio,   vetor_vazio),
            6  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            7  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            8  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            9  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            10 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            11 : combina_media(fontes_1, vet_vazio_int,  vetor_vazio, vetor_vazio, lims_rapido_1, lims_lento_1),
            12 : combina_media(fontes_1, vet_vazio_int,  vetor_vazio, vetor_vazio, lims_rapido_1, lims_lento_1),
            13 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            14 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            15 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            16 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            17 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            18 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            19 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            20 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            21 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            22 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            23 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio)}
        
        self.comb_por_media_temp = {
        #-----------------------------------------------------------------------------------------------------
        # Tipo                  Fonte     Periodo          Offset       Sigma     Limite Rapido   Limite Lento
        #-----------------------------------------------------------------------------------------------------
            0  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            1  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            2  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            3  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            4  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            5  : combina_media(fontes_1, periodos_1,     offsets_1,   sigmas_1,    vetor_vazio,   vetor_vazio),
            6  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            7  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            8  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            9  : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            10 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            11 : combina_media(fontes_1, vet_vazio_int,  vetor_vazio, vetor_vazio, [(x, y) for x, y in zip(lims_rapido_1, lims_lento_1)], vetor_vazio),
            12 : combina_media(fontes_1, vet_vazio_int,  vetor_vazio, vetor_vazio, [(x, y) for x, y in zip(lims_rapido_1, lims_lento_1)], vetor_vazio),
            13 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            14 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            15 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            16 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            17 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            18 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            19 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            20 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            21 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            22 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio),
            23 : combina_media(fontes_1, periodos_1,     vetor_vazio, vetor_vazio, vetor_vazio,   vetor_vazio)}

        for tipo_1 in [11, 12]:
            combs_media_1 = self.comb_por_media_temp[tipo_1]
            for media in combs_media_1:
                media[-1] = media[-2][-1]
                media[-2] = media[-2][-2]


            
    def tipos(self):
        """Retorna array com os possiveis tipos que as médias podem ter"""
        return self.tipos_ma_1
            
    
    def combinacoes_tipos_identicos(self):
        """Retorna array com as combinacoes entre os tipos.\n
        Ex: tipo 0 e tipo 1 -> [[0, 0], [1, 1]]"""
        combinacoes = np.empty(shape = (len(self.tipos_ma_1), 2), dtype = self.tipos_ma_1.dtype)
        combinacoes[:, 0] = self.tipos_ma_1
        combinacoes[:, 1] = self.tipos_ma_1
        return combinacoes
            
    
    def combinacoes_tipos_independentes(self):
        """Retorna array com as combinacoes entre os tipo.\n
        Ex: tipo 0 e tipo 1 -> [[0, 0], [0, 1], [1, 1]]"""
        return np.array(list(combinations_with_replacement(self.tipos_ma_1, 2)))


    def combinacoes_identicas(self, tipo_1, tipo_2):
        media = self.combinacoes_uma_m(tipo_1)

        combinacoes = np.empty(shape = (len(media), 2, 6), dtype = media.dtype)

        combinacoes[:, 0, :] = media # Copia para media 1 e media 2
        combinacoes[:, 1, :] = media
        
        # for i_comb in range(len(media)):
        #     combinacoes[i_comb, 0] = media[i_comb]
        #     combinacoes[i_comb, 1] = media[i_comb]

        return combinacoes

    def combinacoes_mesmo_tipo(self, tipo_1, tipo_2):
        # Faz combinações com tipos iguais, aumenta a velocidade
        # Nao eh MAMA ou FAMA. Tem o atributo periodo
        if tipo_1 != 11 and tipo_1 != 12:
            combs_media_1 = self.comb_por_media_1[tipo_1]
            def qtd_combs_duas():
                contagem_por_periodos_1 = np.empty_like(self.periodos_1)
                for i in range(len(self.periodos_1)):
                    contagem_por_periodos_1[i] = len(list(product(np.array([self.periodos_1[i]]), np.where(self.periodos_1 >= self.periodos_1[i])[0])))
                return int(np.sum(contagem_por_periodos_1)*len(combs_media_1)/len(self.periodos_1))

            qtd = qtd_combs_duas()
            # qtd combinacoes, cada com 2 médias, cada com 6 parametros
            combs_duas_medias = np.empty(qtd*2*6).reshape(qtd, 2, 6)
            
            i = 0 
            for media in combs_media_1:
                periodos_media_2 = self.periodos_1[np.where(self.periodos_1 == media[1])[0][0]:]
                
                for periodo in periodos_media_2:
                    media_2 = np.array(list(media))
                    media_2[1] = periodo
                    combs_duas_medias[i] = np.array([media, media_2])
                    i += 1
                    
            combs_duas_medias = np.array(combs_duas_medias)
        
        else:
            combs_media_1 = self.comb_por_media_temp[tipo_1]
            def qtd_combs_duas():
                contagem_por_lims_rapido_1 = np.empty_like(self.lims_rapido_1)
                for i in range(len(self.lims_rapido_1)):
                    contagem_por_lims_rapido_1[i] = len(list(product(np.array([self.lims_rapido_1[i]]), np.where(self.lims_rapido_1 <= self.lims_rapido_1[i])[0])))
                return int(np.sum(contagem_por_lims_rapido_1)*len(combs_media_1)/len(self.lims_rapido_1))

            qtd = qtd_combs_duas()
            # qtd combinacoes, cada com 2 médias, cada com 6 parametros
            combs_duas_medias = np.empty(qtd*2*6).reshape(qtd, 2, 6)
            
            i = 0
            
            for media in combs_media_1:
                
                l1 = np.where(self.lims_rapido_1 == media[4])[0][0]
                l2 = np.where(self.lims_lento_1 == media[5])[0][0]

                indice = np.maximum(l1, l2)
                
                # Indice do periodo ate o final
                lims_rapido_media_2 = self.lims_rapido_1[indice:]
                lims_lento_media_2 = self.lims_lento_1[indice:]
                
                for lim_rapido, lim_lento in zip(lims_rapido_media_2, lims_lento_media_2):
                    media_2 = np.array(list(media))
                    media_2[4] = lim_rapido
                    media_2[5] = lim_lento
                    
                    combs_duas_medias[i] = np.array([media, media_2])
                    i += 1

            combs_duas_medias = np.array(combs_duas_medias)
    
        return combs_duas_medias

    def combinacoes_independentes(self, tipo_1, tipo_2):
        """Recebe tipos de media e retorna as combinacoes das duas.
        Para iterar, usar a seguinte estrutura:
        for media_1, media_2 in retorno:
            (fon, per, off, sig, lrap, llen) = media_1
            ..."""

        # Faz combinações completas
        if tipo_1 == tipo_2:
            return np.array(list(combinations_with_replacement(self.comb_por_media_1[tipo_1], 2)))
        else:
            return np.array(list(product(self.comb_por_media_1[tipo_1], self.comb_por_media_1[tipo_2])))


    def combinacoes_uma_m(self, tipo):
        """Recebe tipos de media e retorna as combinacoes das duas.
        Para iterar, usar a seguinte estrutura:
        (fon, per, off, sig, lrap, llen) = retorno:
            ..."""
        return self.comb_por_media_1[tipo]

class Media:
    # Quantidade de colunas ocupadas quando uma média é importada pra uma planilha
    COLUNAS_OCUPADAS = 7
    
    def __init__(self) -> None:
        self.tipo = 0
        self.periodos = 13
        self.offset = 0.85
        self.sigma = 6.0
        self.lim_rap = 0.5
        self.lim_lento = 0.05

    def set_tipo(self, tipo:int) -> None:
        """TIPOS\n
        0  - Exponencial\n
        1  - Aritmetica\n
        2  - Ponderada\n
        3  - Hull\n
        4  - Suavizada\n
        5  - Arnaud Legoux (offset, sigma)\n
        6  - Minimos Quadrados (offset, fora de uso)\n
        7  - Exponencial Dupla\n
        8  - Exponencial Tripla\n
        9  - Volume Ponderada\n
        10 - Adaptativa De Kaufman (KAMA)\n
        11 - Adaptativa MESA (MAMA) (lim_rap, lim_lento)\n
        12 - Following MAMA (FAMA) (lim_rap, lim_lento)\n
        13 - Triangular\n
        14 - Mediana Movel\n
        15 - Ajustada ao Volume\n
        16 - Exponencial Zero Lag\n
        17 - Elastica do Volume\n
        18 - Adaptavel Fractal\n
        19 - Adaptavel do Desvio Padrão (DSMA)\n
        20 - Adaptavel do Índice Variável (VIDYA)\n
        21 - Fractal sem limite\n
        22 - Desvio com limite\n
        23 - Ponderada Exponencialmente\n
        """
        self.tipo = tipo
    
    def set_fonte(self, fonte:Fonte) -> None:
        self.fonte = fonte
    
    def set_periodos(self, periodos:int) -> None:
        self.periodos = periodos
    
    def set_offset(self, offset:float = 0.85) -> None:
        self.offset = offset
    
    def set_sigma(self, sigma:float = 6.0) -> None:
        self.sigma = sigma
    
    def set_lim_rap(self, lim_rap:float = 0.5) -> None:
        self.lim_rap = lim_rap
    
    def set_lim_lento(self, lim_lento:float = 0.05) -> None:
        self.lim_lento = lim_lento
    
    def __hash__(self) -> int:
        return hash((self.tipo, self.fonte, self.periodos, self.offset, self.sigma, self.lim_rap, self.lim_lento))
    
    def set_parametros(self, tipo:int, fonte:Fonte, periodos:int, offset:float = 0.85, sigma:float = 6.0, lim_rap:float = 0.5, lim_lento:float = 0.05) -> None:
        self.tipo = tipo
        self.fonte = fonte
        self.periodos = periodos
        self.offset = offset
        self.sigma = sigma
        self.lim_rap = lim_rap
        self.lim_lento = lim_lento
    
    def set_parametros_menos_fonte(self, tipo:int, periodos:int, offset:float = 0.85, sigma:float = 6.0, lim_rap:float = 0.5, lim_lento:float = 0.05) -> None:
        self.tipo = tipo
        self.periodos = periodos
        self.offset = offset
        self.sigma = sigma
        self.lim_rap = lim_rap
        self.lim_lento = lim_lento
        
    def __sub__(self, other:Media) -> np.ndarray:
        return self.valor - other.valor
        
    def __truediv__(self, other:Media) -> np.ndarray:
        return self.valor/other.valor

    def __str__(self) -> str:
        s = f"{nomes_medias[self.tipo]},{self.fonte},{self.periodos},{round(self.offset, 3)},{round(self.sigma, 3)},{round(self.lim_rap, 3)},{round(self.lim_lento, 3)}"
        return s
    
    def __getitem__(self, key:int) -> float:
        return self.valor[key]
    
    def get_media(self) -> np.ndarray:
        return self.valor
        
    def calcula_media(self) -> None:
        # Lenta convergencia: KAMA, Triangular VIDYA
        # Exigem 2 anos de margem de cálculo ou periodos mais curtos
        # Triangular: -15%
        # KAMA, VIDYA: -70%
        # FRAMA: Periodos pares

        if self.tipo == 0:
            self.valor = ema(self.fonte.get_valor(), self.periodos) # numba
            
        elif self.tipo == 1:
            self.valor = sma(self.fonte.get_valor(), self.periodos) # numba
            
        elif self.tipo == 2:
            self.valor = wma(self.fonte.get_valor(), self.periodos) # numba
            
        elif self.tipo == 3:
            self.valor = hma(self.fonte.get_valor(), self.periodos) # numba
        
        elif self.tipo == 4:
            self.valor = smma(self.fonte.get_valor(), self.periodos) # numba
        
        elif self.tipo == 5:
            self.valor = alma(self.fonte.get_valor(), self.periodos, self.offset, self.sigma)
            
        elif self.tipo == 6:
            self.valor = self.lsma(self.fonte.get_valor(), self.periodos)
            
        elif self.tipo == 7:
            self.valor = dema(self.fonte.get_valor(), self.periodos)
            
        elif self.tipo == 8:
            self.valor = tema(self.fonte.get_valor(), self.periodos)
            
        elif self.tipo == 9:
            self.valor = vwma(self.fonte.get_valor(), self.periodos, Fontes.VOLUME)  # numba
        
        elif self.tipo == 10:
            self.valor = kama(self.fonte.get_valor(), self.periodos)
        
        # lim_rap e lim_lento. Variam em [0.036:0.667], default (0.5, 0.05)
        # 0.036 equivale a ema de 54 periodos
        # 0.666 equivale a ema de 2 periodos
        # lim_rap > lim_lento
        # Eles delimitam os limites do alpha da EMA
        # lim_rapido <= 0.99 E lim_lento >= 0.01
        elif self.tipo == 11:
            self.valor = mama(self.fonte.get_valor(), self.lim_rap, self.lim_lento)[0]

        # Mesmo que acima
        elif self.tipo == 12:
            self.valor = mama(self.fonte.get_valor(), self.lim_rap, self.lim_lento)[1]

        elif self.tipo == 13:
            self.valor = trima(self.fonte.get_valor(), self.periodos) # numba
        elif self.tipo == 14:
            self.valor = smm(self.fonte.get_valor(), self.periodos) # numba
        elif self.tipo == 15:
            self.valor = vama(self.fonte.get_valor(), self.periodos, Fontes.VOLUME) # numba
        elif self.tipo == 16:
            self.valor = zlema(self.fonte.get_valor(), self.periodos) # numba
        elif self.tipo == 17:
            self.valor = evwma(self.fonte.get_valor(), self.periodos, Fontes.VOLUME) # numba
        # FRAMA: D é a dimensao fractal dos preços. Varia de 1 até 2;
        # alpha = e^(-4.6*(D - 1))
        elif self.tipo == 18:
            periodos = self.periodos + self.periodos%2
            self.valor = frama(self.fonte.get_valor(), periodos, Fontes.HIGH, Fontes.LOW) # numba
        elif self.tipo == 19:
            self.valor = dsma(self.fonte.get_valor(), self.periodos) # numba
        elif self.tipo == 20:
            # Lenta convergência: 9 - 16 Periodos equivale a ema50
            self.valor = vidya(self.fonte.get_valor(), self.periodos) # numba
        elif self.tipo == 21:
            periodos = self.periodos + self.periodos%2
            self.valor = frama_2(self.fonte.get_valor(), periodos, Fontes.HIGH, Fontes.LOW) # numba
        elif self.tipo == 22:
            self.valor = dsma_2(self.fonte.get_valor(), self.periodos) # numba
        elif self.tipo == 23:
            self.valor = ewma(self.fonte.get_valor(), self.periodos) # numba
            
    
    # Media dos Minimos Quadrados
    def lsma(self, fonte:np.ndarray, periodos:int) -> np.ndarray:
        return linreg(fonte, periodos)

# Janela Movel dos Minimos
@nb.njit
def rolling_min(arr:np.ndarray, periodos:int) -> np.ndarray:
    i_inicial = primer_i_not_nan(arr) + periodos - 1
    out = np.empty_like(arr)
    out[:i_inicial] = np.nan
    lista_ordenada_aux = list(sorted(arr[i_inicial - (periodos - 1):i_inicial + 1]))
    i_final_lista_cheia = periodos - 1
    i_final_lista = periodos - 2
    out[i_inicial] = lista_ordenada_aux[0]
    for i in range(i_inicial + 1, len(arr)):
        lista_ordenada_aux.pop(busca_bin_pop(lista_ordenada_aux, arr[i - periodos], 0, i_final_lista_cheia))
        inclui = arr[i]
        lista_ordenada_aux.insert(busca_bin(lista_ordenada_aux, inclui, 0, i_final_lista), inclui)
        out[i] = lista_ordenada_aux[0]
    return out

# Janela Movel dos Maximos
@nb.njit
def rolling_max(arr:np.ndarray, periodos:int) -> np.ndarray:
    i_inicial = primer_i_not_nan(arr) + periodos - 1
    out = np.empty_like(arr)
    out[:i_inicial] = np.nan
    lista_ordenada_aux = list(sorted(arr[i_inicial - (periodos - 1):i_inicial + 1]))
    i_final_lista_cheia = periodos - 1
    i_final_lista = periodos - 2
    out[i_inicial] = lista_ordenada_aux[-1]
    for i in range(i_inicial + 1, len(arr)):
        lista_ordenada_aux.pop(busca_bin_pop(lista_ordenada_aux, arr[i - periodos], 0, i_final_lista_cheia))
        inclui = arr[i]
        lista_ordenada_aux.insert(busca_bin(lista_ordenada_aux, inclui, 0, i_final_lista), inclui)
        out[i] = lista_ordenada_aux[-1]
    return out

# Janela Movel das somas
@nb.njit(cache = True)
def rolling_sum(arr:np.ndarray, periodos:int) -> np.ndarray:
    out = np.empty_like(arr)
    i_inicial = primer_i_not_nan(arr) + periodos
    out[:i_inicial - 1] = np.nan
    out[i_inicial - 1] = np.sum(arr[i_inicial - periodos:i_inicial])

    for i in range(i_inicial, len(arr)):
        out[i] = out[i - 1] + arr[i] - arr[i - periodos]
    return out

@nb.njit(cache = True)
def get_cci(fonte:np.ndarray, media:np.ndarray, periodos:int) -> np.ndarray:
    distancia_fonte_media = fonte - media
    distancia_media_da_fonte_media = np.empty(len(fonte), np.float64)
    cci = np.empty(len(fonte), np.float64)

    cci[:periodos - 1] = np.nan
    CONSTANTE_CCI = 0.015/periodos # Divide por "periodos" para fazer a media, *0.015 pela formula original
    for i in range(periodos - 1, len(fonte)):
        distancia_media_da_fonte_media[i] = np.abs(distancia_fonte_media[i])
        for j in range(i - 1, i - periodos, -1):
            distancia_media_da_fonte_media[i] += np.abs(fonte[j] - media[i])
        distancia_media_da_fonte_media[i] *= CONSTANTE_CCI

        cci[i] = distancia_fonte_media[i] / distancia_media_da_fonte_media[i]


    return cci

# Janela Movel do Desvio Padrão
@nb.njit(cache = True)
def rolling_std(arr:np.ndarray, periodos:int) -> np.ndarray:
    out = np.empty_like(arr)
    out[:periodos - 1] = np.nan
    periodos_menos_um = periodos - 1
    for i in range(periodos - 1, len(arr)):
        out[i] = np.std(arr[i - periodos_menos_um:i + 1])
    return out


@nb.njit(cache = True)
def change(fonte:np.ndarray) -> np.ndarray:
    change = np.empty_like(fonte)
    change[:1] = np.nan
    change[1:] = (fonte[1:] - fonte[:len(fonte)-1])
    return change



# Encontra o ultimo indice nan de um vetor
@nb.njit
def primer_i_not_nan(arr:np.ndarray) -> int:
    i = 0
    while i < len(arr) and np.isnan(arr[i]):
        i += 1
    return i
#####################################################################################################
######################################### INÍCIO DAS MÉDIAS #########################################
#####################################################################################################

@nb.njit(cache = True)
def ema(fonte:np.ndarray, periodos:int) -> np.ndarray:
    alpha = 2/(periodos + 1)
    out = np.empty_like(fonte)
    i_inicial = primer_i_not_nan(fonte)
    out[:i_inicial] = np.nan
    out[i_inicial] = fonte[i_inicial]

    for i in range(i_inicial + 1, len(fonte)):
        out[i] = fonte[i]*alpha + out[i - 1]*(1 - alpha)

    return out


@nb.njit(cache = True)
def ema_alpha(fonte:np.ndarray, alpha:float) -> np.ndarray:
    out = np.empty_like(fonte)
    i_inicial = max(primer_i_not_nan(fonte), primer_i_not_nan(alpha))
    out[:i_inicial] = np.nan
    out[i_inicial] = fonte[i_inicial]

    for i in range(i_inicial + 1, len(fonte)):
        out[i] = fonte[i]*alpha[i] + out[i - 1]*(1 - alpha[i])

    return out


@nb.njit(cache = True)
def sma(fonte:np.ndarray, periodos:int) -> np.ndarray:
	out = np.empty(len(fonte), np.float64)
	i_inicial = primer_i_not_nan(fonte) + periodos
	out[:i_inicial - 1] = np.nan
	out[i_inicial - 1] = np.sum(fonte[i_inicial - periodos:i_inicial])/periodos
	for i in range(i_inicial, len(fonte)):
		out[i] = out[i - 1] + (fonte[i] - fonte[i - periodos])/periodos
	return out


@nb.njit(cache = True)
def wma(fonte:np.ndarray, periodos:int) -> np.ndarray:
    soma_pesos = soma_pa(periodos)
    out = np.empty_like(fonte)
    i_inicial_not_nan = primer_i_not_nan(fonte)
    i_inicial = i_inicial_not_nan + periodos - 1
    out[:i_inicial] = np.nan
    
    soma = 0.0
    fonte_subtrair_movel = 0.0
    i_aux = 1 - i_inicial_not_nan
    for i in range(i_inicial_not_nan, i_inicial + 1):
        soma += fonte[i]*(i + i_aux)
        fonte_subtrair_movel += fonte[i]
    out[i_inicial] = soma/soma_pesos

    for i in range(i_inicial + 1, len(fonte)):
        soma += fonte[i]*periodos - fonte_subtrair_movel
        out[i] = soma/soma_pesos
        fonte_subtrair_movel += fonte[i] - fonte[i - periodos]

    return out


@nb.njit(cache = True)
def hma(fonte:np.ndarray, periodos:int) -> np.ndarray:
    return wma(2*wma(fonte, periodos//2) - wma(fonte, periodos), floor(sqrt(periodos)))


@nb.njit(cache = True)
def soma_pa(n:int) -> float:
	return (1 + n)*n/2

@nb.njit(cache = True)
def soma_pg(razao:float, n:int) -> float:
    """Soma dos "n" primeiros termos da pg de razao "razao" começada em "1".
    """
    return (razao**n - 1)/(razao - 1)

@nb.njit(cache = True)
def trima(fonte:np.ndarray, periodos:int) -> np.ndarray:
    """"Nao suporta periodo de 1"""

    ceil = np.int64(np.ceil(periodos/2))
    floor = np.int64(np.floor(periodos/2))
    soma_pesos = soma_pa(ceil) + soma_pa(floor)

    out = np.empty_like(fonte)
    i_inicial_not_nan = primer_i_not_nan(fonte)
    i_inicial = i_inicial_not_nan + periodos
    out[:i_inicial - 1] = np.nan
    
    soma = 0.0
    fonte_subtrair_movel = 0.0
    fonte_somar_movel = 0.0

    i_aux = 1 - i_inicial_not_nan
    for i in range(i_inicial_not_nan, i_inicial_not_nan + ceil):
        soma += fonte[i]*(i + i_aux)
        fonte_subtrair_movel += fonte[i]
    i_aux = periodos + i_inicial_not_nan
    for i in range(i_inicial_not_nan + ceil, i_inicial):
        soma += fonte[i]*(i_aux - i)
        fonte_somar_movel += fonte[i]
    
    out[i_inicial - 1] = soma/soma_pesos
    if periodos%2 == 0:
        fonte_somar_movel -= fonte[i_inicial_not_nan + floor]

    ceil_aux = ceil - periodos
    floor_aux = floor + 1 - periodos
    for i in range(i_inicial, len(fonte)):
        fonte_somar_movel += fonte[i]
        soma += fonte_somar_movel - fonte_subtrair_movel
        out[i] = soma/soma_pesos
        fonte_subtrair_movel += fonte[ceil_aux + i] - fonte[i - periodos]
        fonte_somar_movel -= fonte[floor_aux + i]
    return out


@nb.njit(cache = True)
def smma(fonte:np.ndarray, periodos:int) -> np.ndarray:
    alpha = 1/periodos
    out = np.empty_like(fonte)
    i_inicial = primer_i_not_nan(fonte)
    out[:i_inicial] = np.nan
    out[i_inicial] = fonte[i_inicial]

    for i in range(i_inicial + 1, len(fonte)):
        out[i] = fonte[i]*alpha + out[i - 1]*(1 - alpha)

    return out

# Media de Arnaud Legoux
# Nao resulta no primeiro elemento
def alma(fonte:np.ndarray, periodos:int, offset:float, sigma:float) -> np.ndarray: #offset [0: ...] #sigma []
    m = offset * (periodos - 1)
    s = periodos  / sigma
    dss = 2*s**2

    media = np.empty_like(fonte)
    media[:periodos - 1] = np.nan
    wtd = np.exp(-(np.arange(periodos) - m)**2/dss)
    media[periodos - 1:] = np.convolve(wtd[::-1]/np.sum(wtd), fonte, mode = "valid")
    return media


# Media do Volume Ponderada
@nb.njit(cache = True)
def vwma(fonte:np.ndarray, periodos:int, volume:np.ndarray) -> np.ndarray:
    return sma(fonte*volume, periodos)/sma(volume, periodos)

@nb.njit
def busca_bin(vet:list[float], x:float, inicio:int, fim:int) -> int:
    if fim > inicio:
        meio = (inicio + fim)//2
        if x > vet[meio]:
            return busca_bin(vet, x, meio + 1, fim)
        elif x < vet[meio]:
            return busca_bin(vet, x, inicio, meio - 1)
        else:
            return meio
            
    elif inicio == fim:
        if x > vet[inicio]:
            return inicio + 1
        return inicio

    return inicio

@nb.njit
def busca_bin_pop(vet:list[float], x:float, inicio:int, fim:int) -> int:
    if fim > inicio:
        meio = (inicio + fim)//2
        if x == vet[meio]:
            return meio
        elif x > vet[meio]:
            return busca_bin_pop(vet, x, meio + 1, fim)
        else:
            return busca_bin_pop(vet, x, inicio, meio - 1)   
    # inicio == fim:
    return inicio


@nb.njit
def smm_par(arr:np.ndarray, periodos:int) -> np.ndarray:
    i_inicial = primer_i_not_nan(arr) + periodos - 1
    out = np.empty_like(arr)
    out[:i_inicial] = np.nan
    aux = list(sorted(arr[i_inicial - (periodos - 1):i_inicial + 1]))
    i_1 = (len(aux) - 1)//2
    i_2 = len(aux)//2
    out[i_inicial] = (aux[i_1] + aux[i_2])/2
    for i in range(i_inicial + 1, len(arr)):
        aux.pop(busca_bin_pop(aux, arr[i - periodos], 0, len(aux) - 1))
        inclui = arr[i]
        aux.insert(busca_bin(aux, inclui, 0, len(aux) - 1), inclui)
        out[i] = (aux[i_1] + aux[i_2])/2
    return out

@nb.njit
def smm_impar(arr:np.ndarray, periodos:int) -> np.ndarray:
    i_inicial = primer_i_not_nan(arr) + periodos - 1
    out = np.empty_like(arr)
    out[:i_inicial] = np.nan
    aux = list(sorted(arr[i_inicial - (periodos - 1):i_inicial + 1]))
    i_1 = len(aux)//2
    out[i_inicial] = aux[i_1]
    for i in range(i_inicial + 1, len(arr)):
        aux.pop(busca_bin_pop(aux, arr[i - periodos], 0, len(aux) - 1))
        inclui = arr[i]
        aux.insert(busca_bin(aux, inclui, 0, len(aux) - 1), inclui)
        out[i] = aux[i_1]
    return out

@nb.njit
def smm(fonte:np.ndarray, periodos:int) -> np.ndarray:
    if periodos%2:
        return smm_impar(fonte, periodos)
    return smm_par(fonte, periodos)


# Media do Volume Ajustada
@nb.njit(cache = True)
def vama(fonte:np.ndarray, periodos:int, volume:np.ndarray) -> np.ndarray:
    razao = fonte*volume/sma(volume, periodos)
    return rolling_sum(razao*fonte, periodos)/rolling_sum(razao, periodos)


@nb.njit(cache = True)
def zlema(fonte:np.ndarray, periodos:int) -> np.ndarray:
    xLag = (periodos - 1)//2
    out = np.empty_like(fonte)
    out[:xLag] = np.nan
    out[xLag:] = ema(2*fonte[xLag:] - fonte[:-xLag], periodos)
    return out


@nb.njit(cache = True)
def evwma(fonte:np.ndarray, periodos:int, volume:np.ndarray) -> np.ndarray:
    return ema_alpha(fonte, volume/rolling_sum(volume, periodos))

log2 = np.log(2)
# Media Fractal
@nb.njit
def frama(fonte:np.ndarray, periodos:int, high:np.ndarray, low:np.ndarray) -> np.ndarray:

    periodos = periodos + (periodos%2)
    
    meio_per = periodos//2

    n1 = (rolling_max(high, meio_per) - rolling_min(low, meio_per))/meio_per
    
    n2 = np.empty_like(n1)
    n2[:meio_per] = np.nan
    n2[meio_per:] = n1[:-meio_per]

    n3 = (rolling_max(high, periodos) - rolling_min(low, periodos))/periodos

    # calculate fractal dimension
    D = (np.log(n1 + n2) - np.log(n3)) / log2

    alpha = np.exp(-4.6 * (D - 1))
    
    for i in range(len(alpha)):
        if alpha[i] < 0.01:
            alpha[i] = 0.01
        elif alpha[i] > 1:
            alpha[i] = 1

    return ema_alpha(fonte, alpha)

# Media Fractal
@nb.njit # Sem limites no alpha
def frama_2(fonte:np.ndarray, periodos:int, high:np.ndarray, low:np.ndarray) -> np.ndarray:

    periodos = periodos + (periodos%2)
    
    meio_per = periodos//2

    n1 = (rolling_max(high, meio_per) - rolling_min(low, meio_per))/meio_per
    
    n2 = np.empty_like(n1)
    n2[:meio_per] = np.nan
    n2[meio_per:] = n1[:-meio_per]

    n3 = (rolling_max(high, periodos) - rolling_min(low, periodos))/periodos

    # calculate fractal dimension
    D = (np.log(n1 + n2) - np.log(n3)) / log2

    alpha = np.exp(-4.6 * (D - 1))

    return ema_alpha(fonte, alpha)


raiz2_pi = np.sqrt(2)*np.pi
menos_raiz2_pi = -raiz2_pi
@nb.njit
def dsma(fonte:np.ndarray, periodos:int) -> np.ndarray:
    a1 = np.exp(menos_raiz2_pi/(periodos/2))
    b1 = 2*a1*np.cos(raiz2_pi/periodos)
    c2 = b1
    c3 = -a1*a1
    c1 = 1 - c2 - c3
    
    zeros = np.empty_like(fonte)
    zeros[:2] = 0
    zeros[2:] = (fonte[2:] - fonte[:-2])
    
    filt = np.empty_like(fonte)
    filt[:2] = np.nan
    filt[2:] = c1*(zeros[2:] + zeros[1:-1])/2
    # Para nao usar recursividade com "nan", começa a recursividade no primeiro indice valido
    i_inicial = primer_i_not_nan(filt) + 2
    for i in range(i_inicial, len(filt)):
        filt[i] += c2*filt[i - 1] + c3*filt[i - 2]
    scaled_filt = filt/rolling_std(filt, periodos)
    
    alpha = np.abs(scaled_filt)*5/periodos
    
    return ema_alpha(fonte, alpha)

@nb.njit
def dsma_2(fonte:np.ndarray, periodos:int) -> np.ndarray: # Com limites no alpha
    a1 = np.exp(menos_raiz2_pi/(periodos/2))
    b1 = 2*a1*np.cos(raiz2_pi/periodos)
    c2 = b1
    c3 = -a1*a1
    c1 = 1 - c2 - c3
    
    zeros = np.empty_like(fonte)
    zeros[:2] = 0
    zeros[2:] = (fonte[2:] - fonte[:-2])
    
    filt = np.empty_like(fonte)
    filt[:2] = np.nan
    filt[2:] = c1*(zeros[2:] + zeros[1:-1])/2
    # Para nao usar recursividade com "nan", começa a recursividade no primeiro indice valido
    i_inicial = primer_i_not_nan(filt) + 2
    for i in range(i_inicial, len(filt)):
        filt[i] += c2*filt[i - 1] + c3*filt[i - 2]
    scaled_filt = filt/rolling_std(filt, periodos)
    
    alpha = np.empty_like(fonte)
    for i in range(len(scaled_filt)):
        alpha[i] = np.abs(scaled_filt[i])*5/periodos
        if alpha[i] < 0:
            alpha[i] = 0
        elif alpha[i] > 1:
            alpha[i] = 1

    return ema_alpha(fonte, alpha)


@nb.njit(cache = True)
def vidya(fonte:np.ndarray, periodos:int) -> np.ndarray:
    m1 = np.empty_like(fonte)
    m2 = np.empty_like(fonte)

    m1[0] = nan
    m2[0] = nan

    for i in range(1, len(fonte)):
        change = fonte[i] - fonte[i - 1]
        if change > 0:
            m1[i] = change
            m2[i] = 0
        else:
            m1[i] = 0
            m2[i] = -change
    
    sm1 = rolling_sum(m1, periodos)
    sm2 = rolling_sum(m2, periodos)
    #alpha = np.abs((sm1 - sm2)/(sm1 + sm2))*2/(periodos + 1)
    alpha = np.empty_like(sm1)
    alpha_const = 2/(periodos + 1)
    for i in range(len(sm1)):
        soma = sm1[i] + sm2[i]
        if soma:
            alpha[i] = np.abs((sm1[i] - sm2[i])/soma)*alpha_const
        else:
            alpha[i] = fonte[i]*alpha_const
            
    return ema_alpha(fonte, alpha)

@nb.njit(cache = True)
def ewma(fonte:np.ndarray, periodos:int) -> np.ndarray:
    out = np.empty(len(fonte), np.float64)
    i_inicial_not_nan = primer_i_not_nan(fonte)
    i_inicial = i_inicial_not_nan + periodos
    out[:i_inicial_not_nan] = np.nan
    razao = periodos**(1/(periodos - 1))
    razao_acum = 1.0
    soma = 0.0
    for i in range(i_inicial_not_nan, i_inicial - 1):
        soma += fonte[i]*razao_acum
        razao_acum *= razao
        out[i] = np.nan
    
    i = i_inicial - 1
    soma += fonte[i]*periodos
    soma_pesos = soma_pg(razao = razao, n = periodos)
    out[i] = soma/soma_pesos
    
    for i in range(i_inicial, len(fonte)):
        soma = soma/razao + periodos*fonte[i]
        soma_pesos = soma_pesos/razao + periodos
        out[i] = soma/soma_pesos
    return out

# ICHIMOKU

@nb.njit
def donchian(high, low, periodo, Len):
  arr = np.empty(Len)
  arr[:periodo] = np.nan
  for i in range(periodo, Len):
    max = np.max(high[i-periodo+1:i+1])
    min = np.min(low[i-periodo+1:i+1])
    arr[i] = (max+min)/2

  return arr

def tenkan_sen(high, low, periodo):
  Len = len(high)
  return donchian(high, low, periodo, Len)

def kijun_sen(high, low, periodo):
  Len = len(high)
  return donchian(high, low, periodo*3, Len)

def senkou_A(high, low, periodo):
  tenkan = tenkan_sen(high, low, periodo)
  kijun = kijun_sen(high, low, periodo)

  span_A = (np.roll(tenkan,periodo*3) + np.roll(kijun, periodo*3))/2
  span_A[:periodo*3] = np.nan
  
  return span_A

def senkou_B(high, low, periodo):
  Len = len(high)
  span_B = donchian(high, low, periodo*6, Len)
  span_B = np.roll(span_B, periodo*3)
  span_B[:periodo*3] = np.nan

  return span_B

#####################################################################################################
######################################### FIM DAS MÉDIAS #########################################
#####################################################################################################


@nb.njit(cache = True)
def get_forca_motriz(fonte:np.ndarray, media:np.ndarray) -> np.ndarray:
    """Dada por: fonte de hoje - media_ontem. A media eh de "periodos - 1"."""
    out = np.empty(len(fonte), np.float64)
    out[0] = np.nan
    out[1:] = fonte[1:] - media[:-1]
    return out

@nb.njit(cache = True)
def oscilador_redutivo(fonte, peso, periodos = 20) -> np.ndarray:
    # Ex: 4 periodos, fonte = [a b c d]
    # Oscilador = d - (a + b + c)/3
    # O peso é usado para dar mais importancia a eficiencias mais altas. Ele é dividio por 100 para tirar o %
    out = np.abs(fonte)
    out[1:] = fonte[1:]*(1 + peso[1:]/100) - sma(out[:-1], periodos - 1)
    out[0] = np.nan
    return out


@nb.njit(cache = True)
def get_pesos_media_pond(periodos) -> np.ndarray:
    pesos = np.empty(periodos, np.float64)
    pesos[0] = 1.0
    if periodos > 1:
        razao = periodos**(1/(periodos - 1))
        for i in range(1, periodos - 1):
            pesos[i] = pesos[i - 1]*razao
        pesos[-1] = periodos
    return pesos
@nb.njit(cache = True)
def get_soma_pesos_media_pond(periodos) -> np.ndarray:
    if periodos > 1:
        razao = periodos**(1/(periodos - 1))
        return (1 - razao**periodos)/(1 - razao)
    else:
        return 1.0

@nb.njit(cache = True)
def media_harmonica_pond(arr:np.ndarray, pesos:np.ndarray) -> float: #Soma(pesos)/Soma(pesos/fonte)
    # pesos = get_pesos_media_pond(len(arr))
    denominador = 0.0
    for i in range(0, len(arr)):
        if arr[i]:
            denominador += pesos[i]/arr[i]
        else:
            return 0.0 # Media harmonica com valor nulo dá zero
    return get_soma_pesos_media_pond(len(arr))/denominador

@nb.njit(cache = True)
def media_harmonica_pond_descondera_zero(arr:np.ndarray, pesos:np.ndarray) -> float:
    # pesos = get_pesos_media_pond(len(arr))
    numerador = 0.0
    denominador = 0.0
    for i in range(0, len(arr)):
        if arr[i]:
            denominador += pesos[i]/arr[i]
            numerador += pesos[i]
    return numerador/denominador




# Copia do TV abaixo, codigo fora de uso






def media_redutiva(fonte, N) -> np.ndarray:
	out = np.empty(fonte.size, fonte.dtype)
	out[0] = fonte[0]
	out[1] = fonte[1]
	for i in range(2, len(fonte)):
		out[i] = fonte[i]
		soma = 0.0
		for j in range(i - 2, i - N, -1):
			qtd = i - 1 - j
			soma += fonte[j]*soma_pg(N, qtd)/N**qtd
		for j in range(i - N, - 1, -1):
			soma += fonte[j]*soma_pg(N, N - 1)/N**(i - 1 - j)
		out[i] -= soma
	return out


# Para uma janela de tamanho N, gera N - 1 pesos. Eles sao igualmente espaçados e tem
# por inicio o peso 1/N. Sua soma dá 1
def get_pesos_redutiva(N) -> np.ndarray:
	# Formula dada por:
	# (1 + 0*incrm)/N, (1 + 1*incrm)/N, (1 + 2*incrm)/N, ..., (1 + (N - 2)*incrm)/N
	# 1/N, (1/N + 1*incrm/N), (1/N + 2*incrm/N), ...,(1/N + (N - 2)*incrm/N)
	# a1, a1 + incrm_pond, a1 + 2*incrm_pond, ..., a1 + (N - 2)incrm_pond
	pesos = [None]*(N - 1)
	if N == 2:
		pesos[0] = 1.0
		return pesos
	primeiro_peso = 1/N
	incrm = 2/((N - 1)*(N - 2))
	incrm_pond = incrm/N
	
	for i in range(len(pesos)):
		pesos[i] = primeiro_peso + i*incrm_pond
	return pesos
