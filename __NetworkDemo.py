from __Network import Network
from Loss import Loss_CategoricalCrossEntropy
from Optimizers import Optimizer_Adam
from Model import create_data

X, y = create_data(100, 3)

net = Network(2)
net.addDenseLayer(64)
net.addReLU()
net.addDenseLayer(128)
net.addReLU()
net.addDenseLayer(3)
net.addSoftmax()
print(net.getSummary())

loss_function = Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.05, decay=4e-8)

net.train(X, y, 10001, loss_function, optimizer)