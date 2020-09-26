import numpy as np
import scipy.special

class NeuralNet:
    """
    Конструктор класса нейронной сети. На вход: 
     inputNodes - количество входных нейронов
     nodesTuple - количество слоёв скрытых нейронов и их количество в 
        каждом слое задаётся через кортеж
     inputNodes - количество выходных нейронов
     learnRate  - коэффицент обучения нейросети
    """
    def __init__(self, inputNodes, nodesTuple, outNodes, learnRate):     
        self.iNodes = inputNodes
        self.hLayers = len(nodesTuple)
        self.oNodes = outNodes
        self.lRate  = learnRate

        #3-х мерная матрица весов всех связей
        self.linksW = []
        #3-х мерная матрица значений выхода всех узлов
        self.nodes  = []
        #3-х мерная матрица ошибок для каждого узла
        self.errors = []
        
        #Задаём случайные веса всем связям по правилу нормальнго распределения
        #для оптимизации их дальнейшей корректировки

        #для связей между входными нейронами и первым слоем скрытых нейронов
        link = (nodesTuple[0], self.iNodes)
        self.linksW.append( np.random.normal(0.0, nodesTuple[0] ** (-0.5), link) )

        #для дальнейших связей между слоями скрытых нейронах
        bufL = 0
        layer = 0
        for layer in range(self.hLayers - 1):
            bufL += 1
            link = ( nodesTuple[layer + 1], nodesTuple[layer] )
            self.linksW.append( np.random.normal(0.0, nodesTuple[layer + 1] ** (-0.5), link) )

        #для конечной связи между выходными нейронами и последним слоям скрытых
        link = (self.oNodes, nodesTuple[bufL]) if layer == 0 else (self.oNodes, nodesTuple[layer + 1])
        self.linksW.append( np.random.normal(0.0, self.oNodes ** (-0.5), link) )

    """
    Метод для определения квадратичной ошибки, после каждой эпохи
    Чем меньше этот показатель, тем лучше ближе точность к эталону
    """
    def getError(self):
        sumErr = 0
        for er in self.errors[0]:
            sumErr += er * er
        return sumErr
    """
    Метод для обучения нейронной сети на заранее готовых данных
    """
    def train(self, task, asnwer):
        # транспонированные матрицы данных и ответа к ним
        inputs  = np.array(task, ndmin = 2).T
        asnwers = np.array(asnwer, ndmin = 2).T

        self.errors.clear()

        #прогон нейросети по заданным начальным данным
        testOut = self.query(task)

        # Вычисление взвешенных ошибок на каждом узле сети  
        # сначала узнам конечную ошибку, сравнивая с эталоном
        self.errors.append( np.asarray( asnwers - testOut ) )
        
        #обратное распространение ошибки по всем слоям сети
        for layer in range(self.hLayers) :
            self.errors.append( np.dot( self.linksW[self.hLayers - layer].T, \
            self.errors[layer] ) )

        #коррекция весов сети, на основе найденной ошибки
        #методом градиентного спуска нужно искать минимум функции
        #в нашем случае, это будет минимум ошибки на кажом узле
        for layer in range(self.hLayers + 1) :
            self.linksW[self.hLayers - layer] += self.lRate \
            * np.dot( (self.errors[layer] * \
            self.nodes[self.hLayers - layer] * (1 - self.nodes[self.hLayers - layer]) ), \
            ( np.transpose(self.nodes[self.hLayers - layer - 1]) ) \
            if self.hLayers - layer - 1 >= 0 else np.transpose(inputs) )

    """
    Метод запроса к сети, после которого она должна дать ответ
    """
    def query(self, inputTask):
        inputs = np.array(inputTask,ndmin = 2).T
        self.nodes.clear()

        #получение значения выхода на первом слое скрытых нейронов
        #который непосредственно связан со входными значениями
        firstIn = np.dot(self.linksW[0], inputs)
        self.nodes.append( np.asarray( scipy.special.expit(firstIn) ) )

        #последующиее вычисления значений на нейронах по слоям
        for layer in range(self.hLayers):
            out = np.dot(self.linksW[layer + 1], self.nodes[layer])
            self.nodes.append( np.asarray( scipy.special.expit(out) ) )

        #самый последний слой в матрице nodes и будет результирующем 
        #значением которое сформировала сеть
        return self.nodes[-1]

    """
    Метод для вывода всей информации о сети 
    """
    def status(self):
        pass