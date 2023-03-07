from Ativos import *
from Fontes import Fontes
from Indicadores import *
from Medias import qtd_medias
from StrFormatFunctions import join_iteravel


ativos_escolhidos = [DOL, IND]#[DOL, IND, CL, NG, ZN, ZW, DX, KC, CCM, SB]

TAMANHO_RANKING = 10000

ID_DIRETORIO = "BD18.02.2022"

MODO_OPEN = True

QTDS_BARRAS_PARA_BUSCAR_DIVERGS = [20, 30]

QTD_BARRAS_BACKTEST = [30, 60]

# % ACERTO = (Acertos - Erros)/Ocorrencias * 100
ACERTO_MINIMO_TOP = 100
ACERTO_MINIMO_BOT = -100

ANO_I = 2018
ANO_F = 2023


# Os numeros devem ser : 0 <= x < 100
# 0 é o caso extremo, a menor das semelhanças torna duas divergencias como primas
# 100 (que não pode ser escolhido) aceita qualquer combinação, sem distinguir entre primas
SUPERPOSICACAO_DE_PRIMAS = [0, 10, 20, 40]

fontes_open = [1, 22, 43, 71, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 585, 593, 594, 595, 596,
597, 598, 599, 600, 601, 602, 603, 604, 1152, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168,
1169, 1170, 1171]

# 0 - RSI
# 1 - CCI
# 2 - Motriz
# 3 - IFD
# 4 - EFI
# 5 - Momentum
# 6 - CMF
# 7 - Stochastic Original
# 8 - Stochastic Custom
# 9 - UO
indicadores_usados = [1, 8]

# Fonte que eh analisada em conjunto com o indicador para buscar divergencia no movimento
fontes_escolhidas_divergencia = fontes_open

# -------------------- CENARIO MACRO --------------------

# Analise da media movel 20

# 0: Bull : Fonte > Media(Fonte)  Bear : Fonte < Media(Fonte)
# 1: Fonte suavizada eh ignorada na analise macro
papel_da_suavizacao_cenario_macro = [0, 1]
fontes_escolhidas_suavizacao_cenario_macro = [1]
# Tipo da media
tipos_suavizacao_na_suavizacao_cenario_macro = [1]

# Analise de uma fonte a escolha

# 0: Bull : Fonte > Fonte[1]  Bear : Fonte < Fonte[1]
# 1: Fonte extra eh ignorada na analise macro
papel_da_fonte_extra_no_cenario_macro = [0, 1]
fontes_extra_escolhidas_cenario_macro = [22] #range(1,14)

# Analise do indicador

# 0: Usa limites bullish/bearish
# 1: Usa uma media movel 20 do indicador
# 2: Indicador eh ingorado na analise macro
papel_do_indicador_no_cenario_macro = [2]


# -------------------------------------------------------


# RSI
fontes_escolhidas_rsi = [1, 22, 43] #[4]
# Variaveis do Cenario Macro
limites_superior_divergencia_rsi = [50]
limites_inferior_divergencia_rsi = [50]
tipos_suavizacao_rsi = [1] #suavizada


# CCI
fontes_escolhidas_cci = [43] #[4, 7]
# Variaveis do Cenario Macro2
limites_superior_divergencia_cci = [0]
limites_inferior_divergencia_cci = [0]
tipos_suavizacao_cci = [1]


# Forca Motriz
fontes_escolhidas_motriz = [1, 22, 43] #[4]
# Variaveis do Cenario Macro
limites_superior_divergencia_motriz = [0]
limites_inferior_divergencia_motriz = [0]
tipos_suavizacao_motriz = [1]


# IFD
# 139: EFI sem suavização
# 147: OBV
fontes_escolhidas_ifd = [71, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 585, 593, 594, 595, 596,
597, 598, 599, 600, 601, 602, 603, 604, 1152, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168,
1169, 1170, 1171] #+ [139, 147]
# Variaveis do Cenario Macro
tipos_suavizacao_ifd = [1]


# EFI
fontes_escolhidas_efi = [1, 22, 43] # [4]
# Variaveis do Cenario Macro
limites_superior_divergencia_efi = [0]
limites_inferior_divergencia_efi = [0]
tipos_suavizacao_efi = [1]


# Momentum
fontes_escolhidas_momentum = [1, 22, 43] # [4]
# Variaveis do Cenario Macro
limites_superior_divergencia_momentum = [0]
limites_inferior_divergencia_momentum = [0]
tipos_suavizacao_momentum = [1]


# CMF
#  1: Open
#  4: Close *
#  8: HAOpen
# 11: HAClose
fontes_escolhidas_cmf = [1]
#  2: High *
#  9: HAHigh
# 14: TrueHigh
# 16: TrueHighHA
fontes_high_escolhidas_cmf = [2]
#  3: Low *
# 10: HALow
# 15: TrueLow
# 17: TrueLowHA
fontes_low_escolhidas_cmf = [3]

# Variaveis do Cenario Macro
limites_superior_divergencia_cmf = [0]
limites_inferior_divergencia_cmf = [0]
tipos_suavizacao_cmf = [1]


# StochOriginal
#  1: Open
#  4: Close *
#  8: HAOpen
# 11: HAClose
fontes_escolhidas_stoch_original = [1]
#  2: High *
#  9: HAHigh
# 14: TrueHigh
# 16: TrueHighHA
fontes_high_escolhidas_stoch_original = [2]
#  3: Low *
# 10: HALow
# 15: TrueLow
# 17: TrueLowHA
fontes_low_escolhidas_stoch_original = [3]

# Variaveis do Cenario Macro
limites_superior_divergencia_stoch_original = [50]
limites_inferior_divergencia_stoch_original = [50]
tipos_suavizacao_stoch_original = [1]

# StochCustom
fontes_escolhidas_stoch_custom = [1] # [4]
# Variaveis do Cenario Macro
limites_superior_divergencia_stoch_custom = [50]
limites_inferior_divergencia_stoch_custom = [50]
tipos_suavizacao_stoch_custom = [1]


# Ultimate Oscillator
#  1: Open
#  4: Close *
#  8: HAOpen
# 11: HAClose
fontes_escolhidas_uo = [1]
#  2: High
#  9: HAHigh
# 14: TrueHigh *
# 16: TrueHighHA
fontes_high_escolhidas_uo = [2]
#  3: Low
# 10: HALow
# 15: TrueLow *
# 17: TrueLowHA
fontes_low_escolhidas_uo = [3]

# Variaveis do Cenario Macro
limites_superior_divergencia_uo = [50]
limites_inferior_divergencia_uo = [50]
tipos_suavizacao_uo = [1]




# **************************************************************************

# CHECA INDICADORES ESCOLHIDOS
INDICADORES_VALIDOS = {
    0 : "RSI",
    1 : "CCI",
    2 : "Motriz",
    3 : "IFD",
    4 : "EFI",
    5 : "Momentum",
    6 : "CMF",
    7 : "StochOriginal",
    8 : "StochCustom",
    9 : "UO"
}
texto_escolheu_indicador_invalido = ""
for codigo in indicadores_usados:
    if codigo not in INDICADORES_VALIDOS.keys():
        texto_escolheu_indicador_invalido += f"Indicador de codigo '{codigo}' nao existe.\n"
else:
    if texto_escolheu_indicador_invalido:
        texto_escolheu_indicador_invalido += f"\nIndicadores válidos:\n"
        codigos = [f"{cod} : {nome}" for cod, nome in INDICADORES_VALIDOS.items()]
        texto_escolheu_indicador_invalido += "\n".join(codigos) + "\n"
        print(texto_escolheu_indicador_invalido)
        exit()

QTD_GRUPOS_BARRAS_BACKTEST = len(QTD_BARRAS_BACKTEST)
QTD_BARRAS_BACKTEST.sort()
INDICES_FINAIS_BACKTEST = np.array([-i - 1 for i in QTD_BARRAS_BACKTEST])
QTD_BARRAS_BACKTEST_MAX = QTD_BARRAS_BACKTEST[-1]
QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA = QTD_BARRAS_BACKTEST_MAX + 1 # Pega o maior das QTD para ter por referencia

QTD_GRUPOS_BARRAS_PARA_BUSCAR_DIVERGS = len(QTDS_BARRAS_PARA_BUSCAR_DIVERGS)
QTDS_BARRAS_PARA_BUSCAR_DIVERGS.sort()
QTD_TOTAL_BARRAS_PARA_BUSCAR_DIVERGS = QTDS_BARRAS_PARA_BUSCAR_DIVERGS[-1]

if MODO_OPEN:
    str_open = " Open"
else:
    str_open = ""
NOME_DIR_SAIDA = f"{ID_DIRETORIO} B.{join_iteravel(QTD_BARRAS_BACKTEST)} D.{join_iteravel(QTDS_BARRAS_PARA_BUSCAR_DIVERGS)} S.{join_iteravel(SUPERPOSICACAO_DE_PRIMAS)}{str_open}"

fontes : Fontes = None
gabarito_fechamento : np.ndarray = None
indices_gabarito_nao_nulo : np.ndarray = None
qtd_indices_gabarito_nulos : int = None
qtd_indices_gabarito_nulos_arr : list[int] = None
fontes_suavizacao_cenario_macro : Fontes = None
fontes_extra_cenario_macro : Fontes = None
fontes_divergencia : Fontes = None
fontes_rsi : Fontes = None
fontes_cci : Fontes = None
fontes_motriz : Fontes = None
fontes_ifd : Fontes = None
fontes_efi : Fontes = None
fontes_momentum : Fontes = None
fontes_cmf : Fontes = None
fontes_high_cmf : Fontes = None
fontes_low_cmf : Fontes = None
fontes_stoch_original : Fontes = None
fontes_high_stoch_original : Fontes = None
fontes_low_stoch_original : Fontes = None
fontes_stoch_custom : Fontes = None
fontes_uo : Fontes = None
fontes_high_uo : Fontes = None
fontes_low_uo : Fontes = None

@nb.njit(cache = True)
def get_gabarito(opens:np.ndarray, closes:np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Recebe os fechamentos da fonte e faz a análise dos dias positivos, negativos
    e neutros.

    Args:
        fechamento (np.ndarray): Array com os fechamentos do ativo.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: 
        0 - Gabarito: 1 onde for long, -1 short e 0 neutro
        1 - Indices Nao Nulos: Indices de gabarito que não são neutros
        2 - Quantidade Indices Nulos
    """
    if MODO_OPEN: # O modo open considera a abertura de hoje como sendo a referencia. Também, n considera barra[0]
        barras_hoje, barras_ontem = (
        closes[-QTD_BARRAS_BACKTEST_MAX - 1 : -1], opens[-QTD_BARRAS_BACKTEST_MAX - 1 : -1])
    else:
        barras_hoje, barras_ontem = (
        closes[-QTD_BARRAS_BACKTEST_MAX:], closes[-QTD_BARRAS_BACKTEST_MAX - 1 : -1])
        
    gabarito = np.empty(QTD_BARRAS_BACKTEST_MAX, np.int8)
    list_indices_gabarito_nao_nulo = []
    list_qtd_indices_gabarito_nulos = []
    
    i_inicio = -1
    for i_fim in INDICES_FINAIS_BACKTEST:
        indices_gabarito_nao_nulo = []
        for i_barra in range(i_inicio, i_fim, -1):
            hoje = barras_hoje[i_barra]
            ontem = barras_ontem[i_barra]
            
            if hoje > ontem:
                gabarito[i_barra] = 1
                indices_gabarito_nao_nulo.append(i_barra + QTD_BARRAS_BACKTEST_MAX)
            elif hoje < ontem:
                gabarito[i_barra] = -1
                indices_gabarito_nao_nulo.append(i_barra + QTD_BARRAS_BACKTEST_MAX)
            else:
                gabarito[i_barra] = 0
        
        indices_gabarito_nao_nulo = np.array(indices_gabarito_nao_nulo)
        
        barras_avaliadas = (i_inicio - i_fim)
        barras_nao_nulas = len(indices_gabarito_nao_nulo)
        qtd_indices_gabarito_nulos = barras_avaliadas - barras_nao_nulas
        list_indices_gabarito_nao_nulo.append(indices_gabarito_nao_nulo)
        list_qtd_indices_gabarito_nulos.append(qtd_indices_gabarito_nulos)
        i_inicio = i_fim
    
    return gabarito, list_indices_gabarito_nao_nulo, list_qtd_indices_gabarito_nulos

def set_parametros_por_ativo(id_ativo:int) -> None:
    global fontes
    global gabarito_fechamento
    global indices_gabarito_nao_nulo
    global qtd_indices_gabarito_nulos
    global qtd_indices_gabarito_nulos_arr
    global fontes_suavizacao_cenario_macro
    global fontes_extra_cenario_macro
    global fontes_divergencia
    global fontes_rsi
    global fontes_cci
    global fontes_motriz
    global fontes_ifd
    global fontes_efi
    global fontes_momentum
    global fontes_cmf
    global fontes_high_cmf
    global fontes_low_cmf
    global fontes_stoch_original
    global fontes_high_stoch_original
    global fontes_low_stoch_original
    global fontes_stoch_custom
    global fontes_uo
    global fontes_high_uo
    global fontes_low_uo
    
    
    ativo = ativos_escolhidos[id_ativo]
    ativo.set_data(ANO_I, ANO_F)
    fontes = Fontes(ativo)
    assert len(fontes.CLOSE) >= QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, "Erro: Aumente a quantidade de anos de backtest."
    
    gabarito_fechamento, indices_gabarito_nao_nulo, qtd_indices_gabarito_nulos = get_gabarito(
        fontes.OPEN, fontes.CLOSE)
    # Usado para calcular os acertos de cada divergencia. Quando ela acerta uma vez, eh considerado
    # que ela acertou tbm em todos os dias neutros.
    qtd_indices_gabarito_nulos_arr = np.empty(QTD_GRUPOS_BARRAS_BACKTEST, np.int32)
    qtd_indices_gabarito_nulos_arr[0] = qtd_indices_gabarito_nulos[0]#  + 1
    qtd_indices_gabarito_nulos_arr[1:] = qtd_indices_gabarito_nulos[1:]
    
    fontes.checa_fontes_escolhidas(fontes_escolhidas_cmf, fontes_high_escolhidas_cmf, fontes_low_escolhidas_cmf, CMF.NOME_DO_INDICADOR)
    fontes.checa_fontes_escolhidas(fontes_escolhidas_stoch_original, fontes_high_escolhidas_stoch_original, fontes_low_escolhidas_stoch_original, StochasticOrginal.NOME_DO_INDICADOR)
    fontes.checa_fontes_escolhidas(fontes_escolhidas_uo, fontes_high_escolhidas_uo, fontes_low_escolhidas_uo, UltimateOscillator.NOME_DO_INDICADOR)

    # Registra todas as fontes escolhidas
    fontes_suavizacao_cenario_macro = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_suavizacao_cenario_macro)
    fontes_extra_cenario_macro = fontes.get_fontes_escolhidas_por_indice(fontes_extra_escolhidas_cenario_macro)
    fontes_divergencia = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_divergencia)
    
    fontes_rsi = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_rsi)
    
    fontes_cci = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_cci)
    
    fontes_motriz = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_motriz)
    
    fontes_ifd = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_ifd)
    
    fontes_efi = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_efi)
    
    fontes_momentum = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_momentum)
    
    fontes_cmf = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_cmf)
    
    fontes_high_cmf = fontes.get_fontes_escolhidas_por_indice(fontes_high_escolhidas_cmf)
    fontes_low_cmf = fontes.get_fontes_escolhidas_por_indice(fontes_low_escolhidas_cmf)
    
    fontes_stoch_original = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_stoch_original)
    fontes_high_stoch_original = fontes.get_fontes_escolhidas_por_indice(fontes_high_escolhidas_stoch_original)
    fontes_low_stoch_original = fontes.get_fontes_escolhidas_por_indice(fontes_low_escolhidas_stoch_original)
    
    fontes_stoch_custom = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_stoch_custom)
    
    fontes_uo = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_uo)
    fontes_high_uo = fontes.get_fontes_escolhidas_por_indice(fontes_high_escolhidas_uo)
    fontes_low_uo = fontes.get_fontes_escolhidas_por_indice(fontes_low_escolhidas_uo)


CABECALHO = "Suavizacao,#,#,Extra,Indicador,"
CABECALHO_ARQUIVO_JUNTOS = f"Ativo,B/T,Backt,BarrasDiv,Suavizacao,#,#,Extra,Indicador,"
CABECALHO_ARQUIVO_JUNTOS_SEM_PRIMAS = f"Ativo,B/T,Backt,BarrasDiv,SuperP,Suavizacao,#,#,Extra,Indicador,"
CABECALHO_CONTAGEM_VOTOS = f"Arquivo,%Long,%Short,Veredito\n"

# Encontra a quantidade de colunas que o indicador mais "espaçoso" ocupa
COLUNAS_USADAS_POR_INDICADOR_ARQUIVO_SAIDA = get_classe_indicador_por_codigo(indicadores_usados[0]).COLUNAS_OCUPADAS
for codigo_indicador in indicadores_usados[1:]:
    colunas = get_classe_indicador_por_codigo(codigo_indicador).COLUNAS_OCUPADAS
    if colunas > COLUNAS_USADAS_POR_INDICADOR_ARQUIVO_SAIDA:
        COLUNAS_USADAS_POR_INDICADOR_ARQUIVO_SAIDA = colunas
whitespaces = "#,"*(COLUNAS_USADAS_POR_INDICADOR_ARQUIVO_SAIDA - 1)
CABECALHO += whitespaces
CABECALHO_ARQUIVO_JUNTOS += whitespaces
CABECALHO_ARQUIVO_JUNTOS_SEM_PRIMAS += whitespaces

restante = "Limite,#,FonteD,Formato,Categoria,Tipo,Barras,Ocorrencias,Acerto,Indicacao\n"
CABECALHO += restante
CABECALHO_ARQUIVO_JUNTOS += restante
CABECALHO_ARQUIVO_JUNTOS_SEM_PRIMAS += restante