from ArquivoSaida import ArquivoSaida
from Ativos import *
from Fontes import Fontes
from Medias import sma, get_nome_media
from Indicadores import *
import numba as nb
import DivergenciaFunctions as DivergFuncs
from DivergenciaFunctions import main_divergencias_bear, main_divergencias_bull


saida = ArquivoSaida("SaidaDivergencia Gabarito.txt")
saida.writeln("Para pesquisar secoes diferentes, pesquise por \"BUSCA\"")
ativo = DOL


# Fonte que eh analisada em conjunto com o indicador para buscar divergencia no movimento
fontes_escolhidas_divergencia = [4]



# -------------------- CENARIO MACRO --------------------

# Analise do indicador

# 0: Usa limites bullish/bearish
# 1: Usa uma media movel 20 do indicador
# 2: Indicador eh ingorado na analise macro
indicador_no_cenario_macro = [0]

# Analise da media movel 20

# 0: Bull : Fonte > Media(Fonte)  Bear : Fonte < Media(Fonte)
# 1: Fonte suavizada eh ignorada na analise macro
suavizacao_cenario_macro = [0, 1]
fontes_escolhidas_suavizacao_cenario_macro = [1, 4]
tipos_suavizacao_na_suavizacao_cenario_macro = [0, 1]




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
indicadores_usados = [0, 7]


# RSI
fontes_escolhidas_rsi = [4]
# Variaveis do Cenario Macro
limites_superior_divergencia_rsi = [50, 60]
limites_inferior_divergencia_rsi = [40, 50]
tipos_suavizacao_rsi = [0, 1]


# CCI
fontes_escolhidas_cci = [4]
# Variaveis do Cenario Macro
limites_superior_divergencia_cci = [0]
limites_inferior_divergencia_cci = [0]
tipos_suavizacao_cci = [1]


# Forca Motriz
fontes_escolhidas_motriz = [4]
# Variaveis do Cenario Macro
limites_superior_divergencia_motriz = [0]
limites_inferior_divergencia_motriz = [0]
tipos_suavizacao_motriz = [1]


# IFD
# 139: EFI sem suavização
# 147: OBV
fontes_escolhidas_ifd = [139, 147]
# Variaveis do Cenario Macro
tipos_suavizacao_ifd = [1]


# EFI
fontes_escolhidas_efi = [4]
# Variaveis do Cenario Macro
limites_superior_divergencia_efi = [0]
limites_inferior_divergencia_efi = [0]
tipos_suavizacao_efi = [1]


# Momentum
fontes_escolhidas_momentum = [4]
# Variaveis do Cenario Macro
limites_superior_divergencia_momentum = [0]
limites_inferior_divergencia_momentum = [0]
tipos_suavizacao_momentum = [1]


# CMF
#  1: Open
#  4: Close *
#  8: HAOpen
# 11: HAClose
fontes_escolhidas_cmf = [4]
#  2: High *
#  9: HAHigh
# 14: TrueHigh
# 16: TrueHighHA
fontes_high_escolhidas_cmf = [2, 14]
#  3: Low *
# 10: HALow
# 15: TrueLow
# 17: TrueLowHA
fontes_low_escolhidas_cmf = [3, 15]

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
limites_superior_divergencia_stoch_original = [50, 80]
limites_inferior_divergencia_stoch_original = [50]
tipos_suavizacao_stoch_original = [1]

# StochCustom
fontes_escolhidas_stoch_custom = [4]
# Variaveis do Cenario Macro
limites_superior_divergencia_stoch_custom = [50]
limites_inferior_divergencia_stoch_custom = [50]
tipos_suavizacao_stoch_custom = [1]


# Ultimate Oscillator
#  1: Open
#  4: Close *
#  8: HAOpen
# 11: HAClose
fontes_escolhidas_uo = [4]
#  2: High
#  9: HAHigh
# 14: TrueHigh *
# 16: TrueHighHA
fontes_high_escolhidas_uo = [14]
#  3: Low
# 10: HALow
# 15: TrueLow *
# 17: TrueLowHA
fontes_low_escolhidas_uo = [15]

# Variaveis do Cenario Macro
limites_superior_divergencia_uo = [50]
limites_inferior_divergencia_uo = [50]
tipos_suavizacao_uo = [1]



QTD_BARRAS_BACKTEST = 240

ANO_I = 2019
ANO_F = 2022



QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA = QTD_BARRAS_BACKTEST + 1

ativo.set_data(ANO_I, ANO_F)
fontes = Fontes(ativo)
assert len(fontes.CLOSE) >= QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, "Erro: Aumente a quantidade de anos de backtest."

checa_fontes_escolhidas(fontes_escolhidas_cmf, fontes_high_escolhidas_cmf, fontes_low_escolhidas_cmf, CMF.NOME_DO_INDICADOR)
checa_fontes_escolhidas(fontes_escolhidas_stoch_original, fontes_high_escolhidas_stoch_original, fontes_low_escolhidas_stoch_original, StochasticOrginal.NOME_DO_INDICADOR)
checa_fontes_escolhidas(fontes_escolhidas_uo, fontes_high_escolhidas_uo, fontes_low_escolhidas_uo, UltimateOscillator.NOME_DO_INDICADOR)

# Registra todas as fontes escolhidas
fontes_suavizacao_cenario_macro = fontes.get_fontes_escolhidas_por_indice(fontes_escolhidas_suavizacao_cenario_macro)
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

QTD_BARRAS_PARA_BUSCAR_DIVERGS = 10

############
DivergFuncs.dia_foi_bull = np.empty(len(fontes.CLOSE), bool)
DivergFuncs.dia_foi_bull[0] = False
DivergFuncs.dia_foi_bull[1:] = np.where(fontes.CLOSE[1:] > fontes.CLOSE[:-1], True, False)

DivergFuncs.dia_foi_doji_bull = np.empty(len(fontes.CLOSE), bool)
DivergFuncs.dia_foi_doji_bull[0] = False
DivergFuncs.dia_foi_doji_bull[1:] = np.where(
    np.logical_and(
        fontes.CLOSE[1:] == fontes.CLOSE[:-1], fontes.CLOSE[1:] >= fontes.OPEN[1:]
    ), True, False)

############
DivergFuncs.dia_foi_bear = np.empty(len(fontes.CLOSE), bool)
DivergFuncs.dia_foi_bear[0] = False
DivergFuncs.dia_foi_bear[1:] = np.where(fontes.CLOSE[1:] < fontes.CLOSE[:-1], True, False)

DivergFuncs.dia_foi_doji_bear = np.empty(len(fontes.CLOSE), bool)
DivergFuncs.dia_foi_doji_bear[0] = False
DivergFuncs.dia_foi_doji_bear[1:] = np.where(
    np.logical_and(
        fontes.CLOSE[1:] == fontes.CLOSE[:-1], fontes.CLOSE[1:] <= fontes.OPEN[1:]
    ), True, False)


def main_limite_indicador_no_cenario_macro(indicador:Indicador, limites_superior_divergencia:list[float], limites_inferior_divergencia:list[float]):

    for papel_suavizacao_cenario_macro in suavizacao_cenario_macro:
        if papel_suavizacao_cenario_macro == 0:
            media_cenario_macro = Media()
            media_cenario_macro.set_periodos(20)
            for fonte in fontes_suavizacao_cenario_macro:
                media_cenario_macro.set_fonte(fonte)
                for tipo in tipos_suavizacao_na_suavizacao_cenario_macro:
                    media_cenario_macro.set_tipo(tipo)
                    media_cenario_macro.calcula_media()
                    
                    media_suaviz_cenario_macro = media_cenario_macro.get_media()
                    fonte_suaviz_cenario_macro = fonte.get_valor()
                    fonte_acima_da_sua_media = fonte_suaviz_cenario_macro > media_suaviz_cenario_macro
                    fonte_abaixo_da_sua_media = fonte_suaviz_cenario_macro < media_suaviz_cenario_macro



                    # ******
                    for limite_indicador_bull in limites_superior_divergencia:
                        saida.writeln("MACRO FAZ ANALISE DE FONTE X SUAVIZACAO")
                        saida.writeln("MACRO INDICADOR USA LIMITE BULLISH")
                        indicador_acima_limite_bull = indicador > limite_indicador_bull
                        cenario_esta_bull = DivergFuncs.get_cenario_esta_bull(indicador_acima_limite_bull, fonte_acima_da_sua_media)
                        main_divergencias_bear(indicador, cenario_esta_bull, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)
                    
                    for limite_indicador_bear in limites_inferior_divergencia:
                        saida.writeln("MACRO FAZ ANALISE DE FONTE X SUAVIZACAO")
                        saida.writeln("MACRO INDICADOR USA LIMITE BEARISH")
                        indicador_abaixo_limite_bear = indicador < limite_indicador_bear
                        cenario_esta_bear = DivergFuncs.get_cenario_esta_bear(indicador_abaixo_limite_bear, fonte_abaixo_da_sua_media)
                        main_divergencias_bull(indicador, cenario_esta_bear, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)

        elif papel_suavizacao_cenario_macro == 1:
            
            
            # ******
            for limite_indicador_bull in limites_superior_divergencia:
                saida.writeln("MACRO NAO FAZ ANALISE DE FONTE X SUAVIZACAO")
                saida.writeln("MACRO INDICADOR USA LIMITE BULLISH")
                indicador_acima_limite_bull = indicador > limite_indicador_bull
                cenario_esta_bull = DivergFuncs.get_cenario_esta_bull(indicador_acima_limite_bull, True)
                main_divergencias_bear(indicador, cenario_esta_bull, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)
            
            for limite_indicador_bear in limites_inferior_divergencia:
                saida.writeln("MACRO NAO FAZ ANALISE DE FONTE X SUAVIZACAO")
                saida.writeln("MACRO INDICADOR USA LIMITE BEARISH")
                indicador_abaixo_limite_bear = indicador < limite_indicador_bear
                cenario_esta_bear = DivergFuncs.get_cenario_esta_bear(indicador_abaixo_limite_bear, True)
                main_divergencias_bull(indicador, cenario_esta_bear, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)
        else:
            print(f"Erro: Nao existe a suavizacao no cenario macro de codigo '{papel_suavizacao_cenario_macro}'.")
            exit()


def main_suavizacao_do_indicador_no_cenario_macro(indicador:Indicador, tipos_suavizacao_indicador:list[int]):

    for papel_suavizacao_cenario_macro in suavizacao_cenario_macro:
        if papel_suavizacao_cenario_macro == 0:
            media_cenario_macro = Media()
            media_cenario_macro.set_periodos(20)
            for fonte in fontes_suavizacao_cenario_macro:
                media_cenario_macro.set_fonte(fonte)
                for tipo in tipos_suavizacao_na_suavizacao_cenario_macro:
                    media_cenario_macro.set_tipo(tipo)
                    media_cenario_macro.calcula_media()
                    
                    media_suaviz_cenario_macro = media_cenario_macro.get_media()
                    fonte_suaviz_cenario_macro = fonte.get_valor()
                    fonte_acima_da_sua_media = fonte_suaviz_cenario_macro > media_suaviz_cenario_macro
                    fonte_abaixo_da_sua_media = fonte_suaviz_cenario_macro < media_suaviz_cenario_macro

                    # ******
                    ma_indicador = Media()
                    ma_indicador.set_fonte(Fonte(indicador.get_nome_do_indicador(), indicador.get_valor()))
                    ma_indicador.set_periodos(20)
                    for tipo_suavizacao_indicador in tipos_suavizacao_indicador:
                        ma_indicador.set_tipo(tipo_suavizacao_indicador)
                        ma_indicador.calcula_media()

                        saida.writeln("MACRO FAZ ANALISE DE FONTE X SUAVIZACAO")
                        saida.writeln(f"MACRO INDICADOR USA MEDIA \"{get_nome_media(tipo_suavizacao_indicador)}\" 20 periodos")
                        indicador_acima_limite_bull = indicador > ma_indicador.get_media()
                        cenario_esta_bull = DivergFuncs.get_cenario_esta_bull(indicador_acima_limite_bull, fonte_acima_da_sua_media)
                        main_divergencias_bear(indicador, cenario_esta_bull, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)

                        saida.writeln("MACRO FAZ ANALISE DE FONTE X SUAVIZACAO")
                        saida.writeln(f"MACRO INDICADOR USA MEDIA \"{get_nome_media(tipo_suavizacao_indicador)}\" 20 periodos")
                        indicador_abaixo_limite_bear = indicador < ma_indicador.get_media()
                        cenario_esta_bear = DivergFuncs.get_cenario_esta_bear(indicador_abaixo_limite_bear, fonte_abaixo_da_sua_media)
                        main_divergencias_bull(indicador, cenario_esta_bear, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)

        elif papel_suavizacao_cenario_macro == 1:
            # ******
            ma_indicador = Media()
            ma_indicador.set_fonte(Fonte(indicador.get_nome_do_indicador(), indicador.get_valor()))
            ma_indicador.set_periodos(20)
            for tipo_suavizacao_indicador in tipos_suavizacao_indicador:
                ma_indicador.set_tipo(tipo_suavizacao_indicador)
                ma_indicador.calcula_media()

                saida.writeln("MACRO NAO FAZ ANALISE DE FONTE X SUAVIZACAO")
                saida.writeln(f"MACRO INDICADOR USA MEDIA \"{get_nome_media(tipo_suavizacao_indicador)}\" 20 periodos")
                indicador_acima_limite_bull = indicador > ma_indicador.get_media()
                cenario_esta_bull = DivergFuncs.get_cenario_esta_bull(indicador_acima_limite_bull, True)
                main_divergencias_bear(indicador, cenario_esta_bull, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)

                saida.writeln("MACRO NAO FAZ ANALISE DE FONTE X SUAVIZACAO")
                saida.writeln(f"MACRO INDICADOR USA MEDIA \"{get_nome_media(tipo_suavizacao_indicador)}\" 20 periodos")
                indicador_abaixo_limite_bear = indicador < ma_indicador.get_media()
                cenario_esta_bear = DivergFuncs.get_cenario_esta_bear(indicador_abaixo_limite_bear, True)
                main_divergencias_bull(indicador, cenario_esta_bear, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)
        else:
            print(f"Erro: Nao existe a suavizacao no cenario macro de codigo '{papel_suavizacao_cenario_macro}'.")
            exit()
       
def main_nao_analisa_indicador_no_cenario_macro(indicador:Indicador):

    for papel_suavizacao_cenario_macro in suavizacao_cenario_macro:
        if papel_suavizacao_cenario_macro == 0:
            media_cenario_macro = Media()
            media_cenario_macro.set_periodos(20)
            for fonte in fontes_suavizacao_cenario_macro:
                media_cenario_macro.set_fonte(fonte)
                for tipo in tipos_suavizacao_na_suavizacao_cenario_macro:
                    media_cenario_macro.set_tipo(tipo)
                    media_cenario_macro.calcula_media()
                    
                    media_suaviz_cenario_macro = media_cenario_macro.get_media()
                    fonte_suaviz_cenario_macro = fonte.get_valor()
                    fonte_acima_da_sua_media = fonte_suaviz_cenario_macro > media_suaviz_cenario_macro
                    fonte_abaixo_da_sua_media = fonte_suaviz_cenario_macro < media_suaviz_cenario_macro

                    # ******
                    saida.writeln("MACRO FAZ ANALISE DE FONTE X SUAVIZACAO")
                    saida.writeln(f"MACRO SEM USO DO INDICADOR")
                    cenario_esta_bull = DivergFuncs.get_cenario_esta_bull(True, fonte_acima_da_sua_media)
                    main_divergencias_bear(indicador, cenario_esta_bull, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)
                    
                    saida.writeln("MACRO FAZ ANALISE DE FONTE X SUAVIZACAO")
                    saida.writeln(f"MACRO SEM USO DO INDICADOR")
                    cenario_esta_bear = DivergFuncs.get_cenario_esta_bear(True, fonte_abaixo_da_sua_media)
                    main_divergencias_bull(indicador, cenario_esta_bear, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)

        elif papel_suavizacao_cenario_macro == 1:
                    

            # ******
            saida.writeln("MACRO NAO FAZ ANALISE DE FONTE X SUAVIZACAO")
            saida.writeln(f"MACRO SEM USO DO INDICADOR")
            cenario_esta_bull = DivergFuncs.get_cenario_esta_bull(True, True)
            main_divergencias_bear(indicador, cenario_esta_bull, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)
            
            saida.writeln("MACRO NAO FAZ ANALISE DE FONTE X SUAVIZACAO")
            saida.writeln(f"MACRO SEM USO DO INDICADOR")
            cenario_esta_bear = DivergFuncs.get_cenario_esta_bear(True, True)
            main_divergencias_bull(indicador, cenario_esta_bear, fontes_divergencia, QTD_BARRAS_BACKTEST_MAIS_BARRA_AMANHA, QTD_BARRAS_PARA_BUSCAR_DIVERGS)
        else:
            print(f"Erro: Nao existe a suavizacao no cenario macro de codigo '{papel_suavizacao_cenario_macro}'.")
            exit()



def usa_indicador_com_limites_bullish_e_bearish():
    return 0 in indicador_no_cenario_macro
def usa_indicador_com_suavizacao_no_cenario_macro():
    return 1 in indicador_no_cenario_macro
def nao_usa_indicador_na_analise_do_cenario_macro():
    return 2 in indicador_no_cenario_macro

def main_ifd():
    ifd = IFD()
    for fonte_ifd in fontes_ifd:
        ifd.set_fonte(fonte_ifd)
        ifd.calcula_indicador()

        # IFD nao faz uso de limites bullish/bearish
        if usa_indicador_com_limites_bullish_e_bearish():
            pass
        
        if usa_indicador_com_suavizacao_no_cenario_macro():
            main_suavizacao_do_indicador_no_cenario_macro(ifd, tipos_suavizacao_ifd)
        
        if nao_usa_indicador_na_analise_do_cenario_macro():
            main_nao_analisa_indicador_no_cenario_macro(ifd)


def main_rsi():
    rsi = RSI()
    rsi.media_auxiliar.set_tipo(4)
    rsi.media_auxiliar.set_periodos(14)
    for fonte_rsi in fontes_rsi:
        rsi.set_fonte(fonte_rsi)
        rsi.calcula_indicador()

        if usa_indicador_com_limites_bullish_e_bearish():
            main_limite_indicador_no_cenario_macro(rsi, limites_superior_divergencia_rsi, limites_inferior_divergencia_rsi)
        
        if usa_indicador_com_suavizacao_no_cenario_macro():
            main_suavizacao_do_indicador_no_cenario_macro(rsi, tipos_suavizacao_rsi)
        
        if nao_usa_indicador_na_analise_do_cenario_macro():
            main_nao_analisa_indicador_no_cenario_macro(rsi)


def main_cci():
    cci = CCI()
    cci.media_auxiliar.set_tipo(1)
    cci.set_periodos(20)
    for fonte_cci in fontes_cci:
        cci.set_fonte(fonte_cci)
        cci.calcula_indicador()

        if usa_indicador_com_limites_bullish_e_bearish():
            main_limite_indicador_no_cenario_macro(cci, limites_superior_divergencia_cci, limites_inferior_divergencia_cci)
        
        if usa_indicador_com_suavizacao_no_cenario_macro():
            main_suavizacao_do_indicador_no_cenario_macro(cci, tipos_suavizacao_cci)
        
        if nao_usa_indicador_na_analise_do_cenario_macro():
            main_nao_analisa_indicador_no_cenario_macro(cci)

def main_motriz():
    motriz = ForcaMotriz()
    motriz.media_auxiliar.set_tipo(1)
    motriz.set_periodos(20)
    for fonte_motriz in fontes_motriz:
        motriz.media_auxiliar.set_fonte(fonte_motriz)
        motriz.calcula_indicador()

        if usa_indicador_com_limites_bullish_e_bearish():
            main_limite_indicador_no_cenario_macro(motriz, limites_superior_divergencia_motriz, limites_inferior_divergencia_motriz)
        
        if usa_indicador_com_suavizacao_no_cenario_macro():
            main_suavizacao_do_indicador_no_cenario_macro(motriz, tipos_suavizacao_motriz)
        
        if nao_usa_indicador_na_analise_do_cenario_macro():
            main_nao_analisa_indicador_no_cenario_macro(motriz)

def main_efi():
    efi = EFI(fontes.VOLUME)
    efi.media_auxiliar.set_tipo(0)
    efi.media_auxiliar.set_periodos(13)
    for fonte_efi in fontes_efi:
        efi.set_fonte(fonte_efi)
        efi.calcula_indicador()

        if usa_indicador_com_limites_bullish_e_bearish():
            main_limite_indicador_no_cenario_macro(efi, limites_superior_divergencia_efi, limites_inferior_divergencia_efi)
        
        if usa_indicador_com_suavizacao_no_cenario_macro():
            main_suavizacao_do_indicador_no_cenario_macro(efi, tipos_suavizacao_efi)
        
        if nao_usa_indicador_na_analise_do_cenario_macro():
            main_nao_analisa_indicador_no_cenario_macro(efi)

def main_momentum():
    momentum = Momentum()
    momentum.set_periodos(10)
    for fonte_momentum in fontes_momentum:
        momentum.set_fonte(fonte_momentum)
        momentum.calcula_indicador()

        if usa_indicador_com_limites_bullish_e_bearish():
            main_limite_indicador_no_cenario_macro(momentum, limites_superior_divergencia_momentum, limites_inferior_divergencia_momentum)
        
        if usa_indicador_com_suavizacao_no_cenario_macro():
            main_suavizacao_do_indicador_no_cenario_macro(momentum, tipos_suavizacao_momentum)
        
        if nao_usa_indicador_na_analise_do_cenario_macro():
            main_nao_analisa_indicador_no_cenario_macro(momentum)

def main_cmf():
    cmf = CMF(fontes.VOLUME, fontes_high_cmf, fontes_low_cmf)
    cmf.set_periodos(20)
    for i_high in range(len(fontes_high_cmf)):
        cmf.escolhe_fonte_high(i_high)
        
        for i_low in range(len(fontes_low_cmf)):
            cmf.escolhe_fonte_low(i_low)
        
            for fonte in fontes_cmf:
                cmf.set_fonte(fonte)

                cmf.calcula_indicador()

                if usa_indicador_com_limites_bullish_e_bearish():
                    main_limite_indicador_no_cenario_macro(cmf, limites_superior_divergencia_cmf, limites_inferior_divergencia_cmf)
                
                if usa_indicador_com_suavizacao_no_cenario_macro():
                    main_suavizacao_do_indicador_no_cenario_macro(cmf, tipos_suavizacao_cmf)
                
                if nao_usa_indicador_na_analise_do_cenario_macro():
                    main_nao_analisa_indicador_no_cenario_macro(cmf)

def main_stoch_original():
    stoch_original = StochasticOrginal()
    stoch_original.set_periodos(14)
    for fonte_high in fontes_high_stoch_original:
        stoch_original.set_fonte_high(fonte_high)
        
        for fonte_low in fontes_low_stoch_original:
            stoch_original.set_fonte_low(fonte_low)
        
            for fonte in fontes_stoch_original:
                stoch_original.set_fonte(fonte)

                stoch_original.calcula_indicador()

                if usa_indicador_com_limites_bullish_e_bearish():
                    main_limite_indicador_no_cenario_macro(stoch_original, limites_superior_divergencia_stoch_original, limites_inferior_divergencia_stoch_original)
                
                if usa_indicador_com_suavizacao_no_cenario_macro():
                    main_suavizacao_do_indicador_no_cenario_macro(stoch_original, tipos_suavizacao_stoch_original)
                
                if nao_usa_indicador_na_analise_do_cenario_macro():
                    main_nao_analisa_indicador_no_cenario_macro(stoch_original)

def main_stoch_custom():
    stoch_custom = StochasticCustom()
    stoch_custom.set_periodos(14)
    for fonte_stoch_custom in fontes_stoch_custom:
        stoch_custom.set_fonte(fonte_stoch_custom)
        stoch_custom.calcula_indicador()

        if usa_indicador_com_limites_bullish_e_bearish():
            main_limite_indicador_no_cenario_macro(stoch_custom, limites_superior_divergencia_stoch_custom, limites_inferior_divergencia_stoch_custom)
        
        if usa_indicador_com_suavizacao_no_cenario_macro():
            main_suavizacao_do_indicador_no_cenario_macro(stoch_custom, tipos_suavizacao_stoch_custom)
        
        if nao_usa_indicador_na_analise_do_cenario_macro():
            main_nao_analisa_indicador_no_cenario_macro(stoch_custom)


def main_uo():
    uo = UltimateOscillator()
    uo.set_periodos(7)
    uo.media_auxiliar.set_tipo(1)
    for high in fontes_high_uo:
        uo.set_fonte_high(high)
        
        for low in fontes_low_uo:
            uo.set_fonte_low(low)
        
            for fonte in fontes_uo:
                uo.set_fonte(fonte)

                uo.calcula_indicador()

                if usa_indicador_com_limites_bullish_e_bearish():
                    main_limite_indicador_no_cenario_macro(uo, limites_superior_divergencia_uo, limites_inferior_divergencia_uo)
                
                if usa_indicador_com_suavizacao_no_cenario_macro():
                    main_suavizacao_do_indicador_no_cenario_macro(uo, tipos_suavizacao_uo)
                
                if nao_usa_indicador_na_analise_do_cenario_macro():
                    main_nao_analisa_indicador_no_cenario_macro(uo)

FUNCOES_CADA_INDICADOR = {
    0 : main_rsi,
    1 : main_cci,
    2 : main_motriz,
    3 : main_ifd,
    4 : main_efi,
    5 : main_momentum,
    6 : main_cmf,
    7 : main_stoch_original,
    8 : main_stoch_custom,
    9 : main_uo
}
if __name__ == "__main__":
    for codigo_indicador in indicadores_usados:
        funcao = FUNCOES_CADA_INDICADOR[codigo_indicador]
        funcao()