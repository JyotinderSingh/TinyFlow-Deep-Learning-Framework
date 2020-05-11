from __Network import Network

net = Network(2)
net.addDenseLayer(64)
net.addReLU()
net.addDenseLayer(128)
net.addReLU()
net.addDenseLayer(3)
net.addSoftmax()
print(net.getSummary())
