import numpy as np
import scipy.special

class NeuralNet:

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
        link = (nodesTuple[0], self.iNodes)
        self.linksW.append( np.random.normal(0.0, nodesTuple[0] ** (-0.5), link) )

        bufL = 0
        layer = 0
        for layer in range(self.hLayers - 1):
            bufL += 1
            link = ( nodesTuple[layer + 1], nodesTuple[layer] )
            self.linksW.append( np.random.normal(0.0, nodesTuple[layer + 1] ** (-0.5), link) )

        #link = (self.oNodes, nodesTuple[layer + 1]) if layer == 0 else (self.oNodes, nodesTuple[layer])
        link = (self.oNodes, nodesTuple[bufL]) if layer == 0 else (self.oNodes, nodesTuple[layer + 1])
        self.linksW.append( np.random.normal(0.0, self.oNodes ** (-0.5), link) )
    
    def setLinksWeights(self, nodesTuple):
        if len(nodesTuple) < self.hLayers :
            print("Ошибка! Нужно задать количество парцептронов во всех слоях: " + str(self.hLayers))
        else:
            link = (nodesTuple[0], self.iNodes)
            self.linksW.append( np.random.normal(0.0, nodesTuple[0] ** (-0.5), link) )

            layer = 0
            for layer in range(self.hLayers - 1):
                link = ( nodesTuple[layer + 1], nodesTuple[layer] )
                self.linksW.append( np.random.normal(0.0, nodesTuple[layer + 1] ** (-0.5), link) )

            link = (self.oNodes, nodesTuple[layer + 1]) if layer > 1 else (self.oNodes, nodesTuple[layer])
            self.linksW.append( np.random.normal(0.0, self.oNodes ** (-0.5), link) )

    def getError(self):
        sumErr = 0
        for er in self.errors[0]:
            sumErr += er * er
        return sumErr

    def train(self, task, asnwer):
        inputs  = np.array(task, ndmin = 2).T
        asnwers = np.array(asnwer, ndmin = 2).T
        self.errors.clear()

        testOut = self.query(task)    
        self.errors.append( np.asarray( asnwers - testOut ) )
        
        for layer in range(self.hLayers) :
            #обратное распространение ошибки
            self.errors.append( np.dot( self.linksW[self.hLayers - layer].T, \
            self.errors[layer] ) )

        for layer in range(self.hLayers + 1) :
            #коррекция весов
            self.linksW[self.hLayers - layer] += self.lRate \
            * np.dot( (self.errors[layer] * \
            self.nodes[self.hLayers - layer] * (1 - self.nodes[self.hLayers - layer]) ), \
            ( np.transpose(self.nodes[self.hLayers - layer - 1]) ) \
            if self.hLayers - layer - 1 >= 0 else np.transpose(inputs) )

    def query(self, inputTask):
        inputs = np.array(inputTask,ndmin = 2).T
        self.nodes.clear()

        firstIn = np.dot(self.linksW[0], inputs)
        self.nodes.append( np.asarray( scipy.special.expit(firstIn) ) )

        for layer in range(self.hLayers):
            out = np.dot(self.linksW[layer + 1], self.nodes[layer])
            self.nodes.append( np.asarray( scipy.special.expit(out) ) )

        return self.nodes[-1]