import inspect
import numpy as np
import random
import pandas as pd


#Algoritmo básico de GA
class GAalgoritmo_2:
    def __init__(self, parametros):
        self.par = parametros
        self.populacao = []
        self.hist = []
        self.num_dim = self.num_entradas(parametros['funcao'])
    
    def num_entradas(self, funcao):
        """
        Docstring: 
        """
        assinatura = inspect.signature(funcao)
        return len(assinatura.parameters)
    
    def gerar_populacao(self):
        """
        Docstring: 
        """
        populacao = []
        
        for i in range(0, self.par['n_ind']):
            ind = []
            if self.num_dim == 1:
                for j in range(0, self.par['dim']):
                    [inf, sup] = self.par['lim'][j]  # Limite para coordenada X (unidimensional)
                    ind.append(random.uniform(inf,sup))
                X = np.array(ind)
                fitness = self.par['funcao'](X)  # Chamar a função com a coordenada X
                ind.append(fitness)
            
            elif self.num_dim == 2:
                for j in range(0, self.par['dim']):
                    [inf, sup] = self.par['lim'][j]
                    ind.append(random.uniform(inf, sup))
                X = np.array(ind)
                
                ind = []  
                for j in range(0, self.par['dim']):
                    [inf, sup] = self.par['lim'][j]
                    ind.append(random.uniform(inf, sup))
                Y = np.array(ind)
                
                fitness = self.par['funcao'](X, Y)  # Chamar a função com as coordenadas X e Y
                ind.append(sum(fitness))
            
            else:
                # Se a função tem mais de 2 dimensões, você precisará adicionar mais condições aqui
                # para lidar com a chamada correta da função
                pass

            populacao.append(np.array(ind, dtype=object))

        self.populacao = pd.DataFrame(populacao, columns=['X', 'Y', 'Fitness'])
        return self.populacao

    def indexa_populacao(self):
        """
        Docstring: 
        """
        self.populacao = self.populacao.sort_values(by='Fitness', ascending=True)
        self.populacao = self.populacao.reset_index(drop=True)
            
    def crossover(self, id1, id2):
        """
        Docstring: 
        """
        p1 = 0.8*self.populacao.loc[id1][0:2]
        p2 = 0.2*self.populacao.loc[id2][0:2]
        den = 1.0
        
        #p1 = self.populacao.loc[id1][0:2]
        #p2 = self.populacao.loc[id2][0:2]
        #den = 2.0
        
        X = (p1+p2)/den
        return np.array(X)
        
    def mutacao(self, X):
        """
        Docstring: 
        """
        for i in range(0, len(X)):
            X[i] = X[i] + self.par['ft_mut']*random.uniform(-1.0,1.0)
        return X
    
    def evoluir(self):
        """
        Docstring: 
        """
        if len(self.populacao)==0:
            print('Sem populacao')
            return
        n_novos = int(self.par['n_ind']*self.par['ft_cross'])
        n_pop = self.par['n_ind']
        
        #Estrategia de que os melhores sempre estejam no corssover
        lista_novos = []
        for i in range(0, n_novos):
            id1 = i
            #id1 = random.randrange(0, n_novos)
            id2 = random.randrange(n_novos, n_pop)
            novo = self.crossover(id1,id2)
            novo = self.mutacao(novo)
            
            if self.num_dim == 1:
                fitness = self.par['funcao'](novo)  
            elif self.num_dim == 2:
                fitness = self.par['funcao'](novo[0], novo[1])  
            else:
                pass

            novo = np.append(novo, fitness)
            lista_novos.append(novo)
        lista_novos = pd.DataFrame(lista_novos, columns=['X','Y','Fitness'])
        lista_novos = lista_novos.sort_values(by=['Fitness'], ascending=True).reset_index(drop=True)
        return lista_novos
    
    def selecao_natural(self, lista_novos):
        """
        Docstring: 
        """
        n_novos = len(lista_novos)
        n_pop = len(self.populacao)
        for i in range(0, n_novos):
            if lista_novos['Fitness'].iloc[i] < self.populacao['Fitness'].iloc[n_pop-i-1]:
                self.populacao.iloc[n_pop-i-1] = lista_novos.iloc[i]
        self.indexa_populacao()

    def run(self):
        """
        Docstring: 
        """
        self.gerar_populacao()
        self.indexa_populacao()
        self.hist = []
        self.hist.append(self.populacao['Fitness'].iloc[0])
        for i in range(self.par['max_ite']):
            mylist = self.evoluir()
            self.selecao_natural(mylist)
            self.hist.append(self.populacao['Fitness'].iloc[0])
    