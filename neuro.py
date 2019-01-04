import time
import numpy as np
import scipy.special
import neuralNetwork as n

#Параметры нейросети
inputNodes = 784
hiddenNodes = (50, 20)
outNodes = 10
learningRate = 0.5

network = n.NeuralNet(
    inputNodes,
    hiddenNodes,
    outNodes, 
    learningRate
    )

epoch = 0
t = time.time()
print("===обучение===")
for epoch in range(5): 
    trainFile = open('mnist_train.csv', 'r')
    for line in trainFile : 
        indx = int(line[0])
        l = line[2:].split(',')
        data = ( np.asfarray(l) / 255 * 0.99 ) + 0.01
        trueNum = np.zeros(10) + 0.01
        trueNum[indx] = 0.99
        network.train(data, trueNum)
    trainFile.close()
    print("Эпоха #" + str(epoch))
    print("Cуммарная квадратичная ошибка: " + str(network.getError()))


print("===конец обучения===")
print("Время обучения: " + str(time.time() - t) + "\n")

t = 0
f = 0
testFile = open('mnist_test.csv', 'r')
for line in testFile :
    indx = int(line[0])
    l = line[2:].split(',')
    data = ( np.asfarray(l) / 255 * 0.99 ) + 0.01
    answer = network.query(data)
    maxIndx = np.argmax(answer)
    if (maxIndx == indx) :
        t += 1
    else:
        f += 1
print("Успешность: " + str(t / (t + f)))    
testFile.close()    
    

