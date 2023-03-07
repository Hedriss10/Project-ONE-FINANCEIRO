from asyncio import Future
from ArquivoSaida import ArquivoSaida
from Divergencia import Divergencia
from DivergenciaFunctions import amostra_esta_no_conjunto_por_superposicao
from ElosDeRepeticaoDivergencia import RaizDiverg
from ParametrosDivergencias import QTDS_BARRAS_PARA_BUSCAR_DIVERGS, set_parametros_por_ativo, ativos_escolhidos, SUPERPOSICACAO_DE_PRIMAS, CABECALHO, CABECALHO_ARQUIVO_JUNTOS, CABECALHO_ARQUIVO_JUNTOS_SEM_PRIMAS, CABECALHO_CONTAGEM_VOTOS, indicadores_usados, QTD_BARRAS_BACKTEST, NOME_DIR_SAIDA, QTD_GRUPOS_BARRAS_BACKTEST, QTD_GRUPOS_BARRAS_PARA_BUSCAR_DIVERGS
from FuncoesMultprocessamento import divide_lacos_aninhados_por_processo
import concurrent.futures
import traceback
from time import perf_counter
from datetime import timedelta, datetime
from DirFunctions import force_mkdir
from RankingDinamicoSimples import RankingDinamico


def main(indices_lacos_particionados:list[int], i_process:int, nome_dir_saida:str
         ) -> dict[int, tuple[list[list[RankingDinamico]], list[list[RankingDinamico]]]]:
    # [ativo][maiores | menores][grupo barra divergencia][grupo barra backtest]
    total_reps = len(indices_lacos_particionados)

    try:
        reps_completadas = 0
        rankings_dinamicos_por_ativo : dict[int, tuple[list[list[RankingDinamico]], list[list[RankingDinamico]]]] = {}
        def cria_nova_raiz(i_indicador:int) -> RaizDiverg:
            raiz_lacos_rep  = RaizDiverg()
            
            raiz_lacos_rep.elo_3_indicador_no_cenario_macro.set_atributos_necessarios_elo()
            raiz_lacos_rep.elo_2_fonte_extra_no_cenario_macro.set_atributos_necessarios_elo()
            raiz_lacos_rep.elo_1_suavizacao_no_cenario_macro.set_atributos_necessarios_elo()
            set_atributos_elo_0(raiz_lacos_rep, i_indicador)
            return raiz_lacos_rep
        
        def set_atributos_elo_0(raiz_lacos_rep:RaizDiverg, i_indicador:int) -> None:
            raiz_lacos_rep.elo_0_calculo_indicador.set_codigos_lacos_escolhidas_usuario(
                                                            [indicadores_usados[i_indicador]])
            raiz_lacos_rep.elo_0_calculo_indicador.set_atributos_necessarios_elo()
        
        # Vai ser processado uma serie de ativos e indicadores. Começa inicializando
        # as variaveis pro primeiro ativo e pimeiro indicador
        i_ativo_ant = indices_lacos_particionados[0][0]
        i_indicador = indices_lacos_particionados[0][1]
        set_parametros_por_ativo(i_ativo_ant)
        raiz_lacos_rep = cria_nova_raiz(i_indicador)
        
        def registra_rankings_do_ativo(i_ativo_ant:int, raiz_lacos_rep:RaizDiverg) -> None:
                mem = raiz_lacos_rep.mem_compartilhada_elos
                
                ranking_maiores, ranking_menores = (
                    mem.ranking_maiores, mem.ranking_menores
                )
                
                rankings_dinamicos_por_ativo[i_ativo_ant] = (
                        ranking_maiores, ranking_menores
                )
        
        def mudou_o_ativo(i_ativo_ant:int, i_ativo:int) -> None:
            return i_ativo != i_ativo_ant
                    
        for indices_lacos_atual in indices_lacos_particionados:
            i_ativo = indices_lacos_atual[0]
            i_indicador = indices_lacos_atual[1]
            tempo_inicial_execucao = perf_counter()
            set_parametros_por_ativo(i_ativo)
            
            # Caso esteja num novo ativo, então registra os rankings
            # dinamicos feitos e cria uma nova raiz. Do contrário, 
            # prepara o Elo 0 para o proximo indicador.
            if mudou_o_ativo(i_ativo_ant, i_ativo):
                registra_rankings_do_ativo(i_ativo_ant, raiz_lacos_rep)
                raiz_lacos_rep = cria_nova_raiz(i_indicador)
                i_ativo_ant = i_ativo
            else:
                set_atributos_elo_0(raiz_lacos_rep, i_indicador)
                
            raiz_lacos_rep.main()
            
            reps_completadas += 1
            tempo_execucao = perf_counter() - tempo_inicial_execucao
            qtd_reps_restantes = total_reps - reps_completadas
            if qtd_reps_restantes > 0:
                percentual_concluido = (1 - qtd_reps_restantes/total_reps)*100
                tempo_estimado = tempo_execucao*qtd_reps_restantes
                print(f"Processo {i_process}\n"
                        f"{percentual_concluido:.1f}% Concluído\n"
                        f"Tempo para conclusão - {timedelta(seconds = round(tempo_estimado))}.\n")
            
        registra_rankings_do_ativo(i_ativo_ant, raiz_lacos_rep)
        print(f"Processo {i_process} Concluído\n")
        return rankings_dinamicos_por_ativo
    
    except:
        print(f"Erro: Falha na execução no Processo {i_process}, responsável pelo range{tuple(indices_lacos_particionados)}.")
        traceback.print_exc()
        print()
        exit()

if __name__ == "__main__":
    data = datetime.now().strftime("%d.%m.%y %H.%M")
    nome_dir_saida = fr"Resultados\Divergencias\{NOME_DIR_SAIDA} {data}"
    force_mkdir(nome_dir_saida)
    
    def get_nome_arquivo_normal(i_ativo:int, qtd_barras_diverg:int, qtd_barras_backtest:int,
    tipo_do_ranking:str) -> bool:
        """
        i_ativo : Indice do ativo analisado
        qtd_barras_backtest :  Numero de barras avaliadas
        tipo_do_ranking : "Top" ou "Bot"
        """
        nome_arquivo = f"{ativos_escolhidos[i_ativo].nome} {tipo_do_ranking} B{qtd_barras_backtest} D{qtd_barras_diverg}.txt"
        return f"{nome_arquivo}"
    
    def get_nome_arquivo_sem_primas(i_ativo:int, qtd_barras_diverg:int, qtd_barras_backtest:int,
    tipo_do_ranking:str, superposicao_aceitavel:int, analisa_barra_pivo:bool) -> bool:
        """
        Mesmo que acima, com a adição:
        superposicao_aceitavel : % de superposição permitido para duas divergecnasi não serem primas.
        analisa_barra_pivo : Sem Primas com referencia na barra pivo (True) ou na barra de apoio (False)
        """
        str_barra = "Pivo" if analisa_barra_pivo else "Apoio"
        nome_arquivo = f"{ativos_escolhidos[i_ativo].nome} {tipo_do_ranking} B{qtd_barras_backtest} D{qtd_barras_diverg} SP{superposicao_aceitavel} {str_barra}.txt"
        return f"{nome_arquivo}"
    
    
    tempo_inicial_execucao = perf_counter()
    
    args_lacos_divididos_por_processo = divide_lacos_aninhados_por_processo(
        ativos_escolhidos, indicadores_usados)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures : list[Future] = []

        for i_process, indices_lacos_particionados in enumerate(args_lacos_divididos_por_processo):
            
            f = executor.submit(main, indices_lacos_particionados, i_process, nome_dir_saida)
            
            futures.append(f)
        
        concurrent.futures.wait(futures)
        
        rankings_dinamicos_por_ativo:dict[int, tuple[list[list[RankingDinamico]], list[list[RankingDinamico]]]]
        rankings_dinamicos_por_ativo_add:dict[int, tuple[list[list[RankingDinamico]], list[list[RankingDinamico]]]]
        
        # Junta os rankings de diferentes processamentos
        rankings_dinamicos_por_ativo = futures[0].result()
        # [ativo][maiores | menores][grupo barra divergencia][grupo barra backtest]
        for f in futures[1:]:
            rankings_dinamicos_por_ativo_add = f.result()
            
            for i_ativo, (list_list_ranking_maiores_add, list_list_ranking_menores_add) in (
                rankings_dinamicos_por_ativo_add.items()):
                # Adiciona aos ranking já existentes
                if i_ativo in rankings_dinamicos_por_ativo.keys():
                    list_list_ranking_maiores, list_list_ranking_menores = rankings_dinamicos_por_ativo[i_ativo]
                    
                    for i in range(QTD_GRUPOS_BARRAS_PARA_BUSCAR_DIVERGS):
                        list_ranking_maiores = list_list_ranking_maiores[i]
                        list_ranking_menores = list_list_ranking_menores[i]
                        list_ranking_maiores_add = list_list_ranking_maiores_add[i]
                        list_ranking_menores_add = list_list_ranking_menores_add[i]

                        for j in range(QTD_GRUPOS_BARRAS_BACKTEST):
                            list_ranking_maiores[j].extend_ranking(list_ranking_maiores_add[j])
                            list_ranking_menores[j].extend_ranking(list_ranking_menores_add[j])
                else:
                    rankings_dinamicos_por_ativo[i_ativo] = list_list_ranking_maiores_add, list_list_ranking_menores_add
    
    # Junta Rankings de diferentes Grupos de Barras Divergencia
    # Por ex, caso seja escolhido as divergencias de [20, 30], entao o ranking 1 vai conter apenas as divergencias 20
    # e o ranking 2 vai conter apenas as divergencias de 21 a 30 barras atras. Para corrigir isso, inclui o ranking 1
    # no ranking 2. Esse procedimento eh sequencial
    for list_list_ranking_maiores, list_list_ranking_menores in (
        rankings_dinamicos_por_ativo.values()):

        for j in range(QTD_GRUPOS_BARRAS_BACKTEST):
            
            for i in range(QTD_GRUPOS_BARRAS_PARA_BUSCAR_DIVERGS - 1):
                ranking_maiores = list_list_ranking_maiores[i][j]
                ranking_maiores_prox = list_list_ranking_maiores[i + 1][j]
                ranking_maiores_prox.extend_ranking(ranking_maiores)
                
                ranking_menores = list_list_ranking_menores[i][j]
                ranking_menores_prox = list_list_ranking_menores[i + 1][j]
                ranking_menores_prox.extend_ranking(ranking_menores)


    class Votos:
        def __init__(self, arquivo:str) -> None:
            self.arquivo = arquivo
            self.long = 0
            self.short = 0
        
        def add_voto(self, diverg:Divergencia) -> None:
            if diverg.indicacao == "LONG":
                self.long += 1
            elif diverg.indicacao == "SHORT":
                self.short += 1
        
        def __str__(self) -> str:
            total = self.long + self.short
            if total:
                str_long  = f"{self.long/total*100:.0f}"
                str_short = f"{self.short/total*100:.0f}"
            else:
                str_long  = "nan"
                str_short = "nan"
            
            veredito = self.long - self.short
            if veredito > 0:
                str_veredito = "LONG"
            elif veredito < 0:
                str_veredito = "SHORT"
            else:
                str_veredito = "NEUTRO"
            return f"{self.arquivo},{str_long},{str_short},{str_veredito}"


    # IMPRIME RANKINGS
    
    saida_juntos_normal = open(fr"{nome_dir_saida}\Juntos Com Primas.txt", "w")
    saida_juntos_normal.write(f"{CABECALHO_ARQUIVO_JUNTOS}")
    votos_juntos_normal = Votos("Juntos Com Primas.txt")
    
    saida_juntos_sp_pivo = open(fr"{nome_dir_saida}\Juntos Sem Primas Pivo.txt", "w")
    saida_juntos_sp_pivo.write(f"{CABECALHO_ARQUIVO_JUNTOS_SEM_PRIMAS}")
    votos_juntos_sp_pivo = Votos("Juntos Sem Primas Pivo.txt")
    
    saida_juntos_sp_apoio = open(fr"{nome_dir_saida}\Juntos Sem Primas Apoio.txt", "w")
    saida_juntos_sp_apoio.write(f"{CABECALHO_ARQUIVO_JUNTOS_SEM_PRIMAS}")
    votos_juntos_sp_apoio = Votos("Juntos Sem Primas Apoio.txt")

    saida_contagem_de_votos = open(fr"{nome_dir_saida}\Contagem Votos.txt", "w")
    saida_contagem_de_votos.write(f"{CABECALHO_CONTAGEM_VOTOS}")
    
    diverg:Divergencia
    for i_ativo, (list_list_ranking_maiores, list_list_ranking_menores) in (
        rankings_dinamicos_por_ativo.items()):
        for i, qtd_barras_diverg in enumerate(QTDS_BARRAS_PARA_BUSCAR_DIVERGS):
            list_ranking_maiores = list_list_ranking_maiores[i]
            list_ranking_menores = list_list_ranking_menores[i]

            for tipo_do_ranking in ["Top", "Bot"]:
                if tipo_do_ranking == "Top":
                    list_ranking = list_ranking_maiores
                else:
                    list_ranking = list_ranking_menores

                for ranking, qtd_barras_backtest in zip(list_ranking, QTD_BARRAS_BACKTEST):
                    ranking.fecha_ranking()
                    ranking_fechado = ranking.get_ranking()


                    superposicao_aceitavel = SUPERPOSICACAO_DE_PRIMAS[0]
                    # Com primas
                    nome_arquivo = get_nome_arquivo_normal(i_ativo, qtd_barras_diverg, qtd_barras_backtest, tipo_do_ranking)
                    saida_normal = open(fr"{nome_dir_saida}\{nome_arquivo}", "w")
                    saida_normal.write(CABECALHO)
                    votos_normal = Votos(nome_arquivo)
                    # Sem primas: Amostra é a barra pivo
                    nome_arquivo = get_nome_arquivo_sem_primas(i_ativo, qtd_barras_diverg, qtd_barras_backtest, tipo_do_ranking, superposicao_aceitavel, True)
                    saida_sp_pivo = open(fr"{nome_dir_saida}\{nome_arquivo}", "w")
                    saida_sp_pivo.write(CABECALHO)
                    votos_sp_pivo = Votos(nome_arquivo)
                    # Sem primas: Amostra é a barra de apoio
                    nome_arquivo = get_nome_arquivo_sem_primas(i_ativo, qtd_barras_diverg, qtd_barras_backtest, tipo_do_ranking, superposicao_aceitavel, False)
                    saida_sp_apoio = open(fr"{nome_dir_saida}\{nome_arquivo}", "w")
                    saida_sp_apoio.write(CABECALHO)
                    votos_sp_apoio = Votos(nome_arquivo)

                    texto_inicio_juntos = f"{ativos_escolhidos[i_ativo].nome},{tipo_do_ranking},{qtd_barras_backtest},{qtd_barras_diverg}"
                    texto_inicio_juntos_sp = f"{texto_inicio_juntos},{superposicao_aceitavel}"

                    if len(ranking_fechado) == 0:
                        continue
                    
                    diverg = ranking_fechado[0]
                    diverg.set_i_barras_apoio_com_ocorrencias()
                    barras_incluidas_sp_pivo = diverg.i_barras_com_ocorrencias.copy()
                    barras_incluidas_sp_apoio = diverg.i_barras_apoio_com_ocorrencias.copy()
                    
                    votos_normal.add_voto(diverg)
                    votos_juntos_normal.add_voto(diverg)
                    votos_sp_pivo.add_voto(diverg)
                    votos_juntos_sp_pivo.add_voto(diverg)
                    votos_sp_apoio.add_voto(diverg)
                    votos_juntos_sp_apoio.add_voto(diverg)
                    
                    saida_normal.write(f"{diverg}\n")
                    
                    saida_juntos_normal.write(f"{texto_inicio_juntos},{diverg}\n")
                    saida_juntos_sp_pivo.write(f"{texto_inicio_juntos_sp},{diverg}\n")
                    saida_juntos_sp_apoio.write(f"{texto_inicio_juntos_sp},{diverg}\n")

                    saida_sp_pivo.write(f"{diverg}\n")
                    saida_sp_apoio.write(f"{diverg}\n")

                    # ranking_maiores.fecha_ranking()
                    for diverg in ranking_fechado[1:]:
                        votos_normal.add_voto(diverg)
                        votos_juntos_normal.add_voto(diverg)

                        saida_normal.write(f"{diverg}\n")
                        saida_juntos_normal.write(f"{texto_inicio_juntos},{diverg}\n")
                        
                        if amostra_esta_no_conjunto_por_superposicao(diverg.i_barras_com_ocorrencias, barras_incluidas_sp_pivo, superposicao_aceitavel):
                            votos_sp_pivo.add_voto(diverg)
                            votos_juntos_sp_pivo.add_voto(diverg)

                            saida_sp_pivo.write(f"{diverg}\n")
                            saida_juntos_sp_pivo.write(f"{texto_inicio_juntos_sp},{diverg}\n")
                    
                        diverg.set_i_barras_apoio_com_ocorrencias()
                        if amostra_esta_no_conjunto_por_superposicao(diverg.i_barras_apoio_com_ocorrencias, barras_incluidas_sp_apoio, superposicao_aceitavel):
                            votos_sp_apoio.add_voto(diverg)
                            votos_juntos_sp_apoio.add_voto(diverg)

                            saida_sp_apoio.write(f"{diverg}\n")
                            saida_juntos_sp_apoio.write(f"{texto_inicio_juntos_sp},{diverg}\n")


                    saida_normal.close()
                    saida_sp_pivo.close()
                    saida_sp_apoio.close()
                    
                    saida_contagem_de_votos.write(f"{votos_normal}\n")
                    saida_contagem_de_votos.write(f"{votos_sp_pivo}\n")
                    saida_contagem_de_votos.write(f"{votos_sp_apoio}\n")

                    for superposicao_aceitavel in SUPERPOSICACAO_DE_PRIMAS[1:]:
                        # Sem primas: Amostra é a barra pivo
                        nome_arquivo = get_nome_arquivo_sem_primas(i_ativo, qtd_barras_diverg, qtd_barras_backtest, tipo_do_ranking, superposicao_aceitavel, True)
                        saida_sp_pivo = open(fr"{nome_dir_saida}\{nome_arquivo}", "w")
                        saida_sp_pivo.write(CABECALHO)
                        votos_sp_pivo = Votos(nome_arquivo)
                        # Sem primas: Amostra é a barra de apoio
                        nome_arquivo = get_nome_arquivo_sem_primas(i_ativo, qtd_barras_diverg, qtd_barras_backtest, tipo_do_ranking, superposicao_aceitavel, False)
                        saida_sp_apoio = open(fr"{nome_dir_saida}\{nome_arquivo}", "w")
                        saida_sp_apoio.write(CABECALHO)
                        votos_sp_apoio = Votos(nome_arquivo)

                        texto_inicio_juntos = f"{ativos_escolhidos[i_ativo].nome},{tipo_do_ranking},{qtd_barras_backtest}"
                        texto_inicio_juntos_sp = f"{texto_inicio_juntos},{superposicao_aceitavel}"

                        diverg = ranking_fechado[0]
                        barras_incluidas_sp_pivo = diverg.i_barras_com_ocorrencias.copy()
                        barras_incluidas_sp_apoio = diverg.i_barras_apoio_com_ocorrencias.copy()
                        
                        votos_sp_pivo.add_voto(diverg)
                        votos_juntos_sp_pivo.add_voto(diverg)
                        votos_sp_apoio.add_voto(diverg)
                        votos_juntos_sp_apoio.add_voto(diverg)
                        
                        saida_juntos_sp_pivo.write(f"{texto_inicio_juntos_sp},{diverg}\n")
                        saida_juntos_sp_apoio.write(f"{texto_inicio_juntos_sp},{diverg}\n")

                        saida_sp_pivo.write(f"{diverg}\n")
                        saida_sp_apoio.write(f"{diverg}\n")

                        for diverg in ranking_fechado[1:]:
                            
                            if amostra_esta_no_conjunto_por_superposicao(diverg.i_barras_com_ocorrencias, barras_incluidas_sp_pivo, superposicao_aceitavel):
                                votos_sp_pivo.add_voto(diverg)
                                votos_juntos_sp_pivo.add_voto(diverg)

                                saida_sp_pivo.write(f"{diverg}\n")
                                saida_juntos_sp_pivo.write(f"{texto_inicio_juntos_sp},{diverg}\n")
                        
                            if amostra_esta_no_conjunto_por_superposicao(diverg.i_barras_apoio_com_ocorrencias, barras_incluidas_sp_apoio, superposicao_aceitavel):
                                votos_sp_apoio.add_voto(diverg)
                                votos_juntos_sp_apoio.add_voto(diverg)

                                saida_sp_apoio.write(f"{diverg}\n")
                                saida_juntos_sp_apoio.write(f"{texto_inicio_juntos_sp},{diverg}\n")


                        saida_sp_pivo.close()
                        saida_sp_apoio.close()
                    
                        saida_contagem_de_votos.write(f"{votos_sp_pivo}\n")
                        saida_contagem_de_votos.write(f"{votos_sp_apoio}\n")


    saida_juntos_normal.close()
    saida_juntos_sp_pivo.close()
    saida_juntos_sp_apoio.close()
    
    saida_contagem_de_votos.write(f"{votos_juntos_normal}\n")
    saida_contagem_de_votos.write(f"{votos_juntos_sp_pivo}\n")
    saida_contagem_de_votos.write(f"{votos_juntos_sp_apoio}\n")


    tempo_execucao = perf_counter() - tempo_inicial_execucao
    h = tempo_execucao//3600
    m = (tempo_execucao - h*3600)//60
    s = tempo_execucao%60
    print(f"Concluido em {h:.0f} horas {m:.0f} minutos {s:.0f} segundos")