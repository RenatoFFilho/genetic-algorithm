import fobs
from GAalgoritmo_2 import GAalgoritmo_2



# Exempla para a função de Ackley
# Corrigido com o novo parâmetro 'num_dimensoes_fitness'
parametros = {
    'funcao': fobs.mishra_bird_constrained,
    'lim': fobs.limit_mishra_bird_constrained,
    'dim': 2,  # Número de dimensões das coordenadas do espaço de busca
    'n_ind': 100,
    'ft_mut': 0.1,
    'ft_cross': 0.8,
    'max_ite': 100,
    'num_dimensoes_fitness': 1,  # Número de dimensões da função de aptidão
    'tipo_mut': 'shrink'
}

ga = GAalgoritmo_2(parametros)
ga.gerar_populacao()
ga.indexa_populacao()
ga.evoluir()
ga.run()
print(ga.populacao)

