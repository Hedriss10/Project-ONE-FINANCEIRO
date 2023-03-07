from Comparavel import ComparavelSimples

# Armazena as informações de rendimento para cada ano
class Divergencia(ComparavelSimples):
    """Para haver divergencia eh analisada duas barras. A barra de hoje é chamada de
    "barra pivo", enquanto que a barra antiga é chamada de "barra de apoio".

    Args:
        ComparavelSimples (_type_): _description_
    """
    def set_qtd_barras(self, qtd_barras:int) -> None:
        """A divergencia eh analisada tendo por apoio a barrra de hoje
        e uma barra a "qtd_barras" atras.
        """

        self.qtd_barras = qtd_barras
    
    def set_i_barras_com_ocorrencias(self, i_barras_com_ocorrencias:list[int]) -> None:
        self.i_barras_com_ocorrencias = i_barras_com_ocorrencias
        
    def set_i_barras_apoio_com_ocorrencias(self) -> list[int]:
        self.i_barras_apoio_com_ocorrencias = [i - self.qtd_barras for i in self.i_barras_com_ocorrencias]
    
    def armazena_indicacao(self, indicacao:str) -> None:
        self.indicacao = indicacao