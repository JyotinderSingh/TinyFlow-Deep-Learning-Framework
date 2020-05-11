from __Network import Network

net = Network(2)
net.addDenseLayer(64)
net.addDenseLayer(64)
net.addDenseLayer(3)
print(net.getSummary())
