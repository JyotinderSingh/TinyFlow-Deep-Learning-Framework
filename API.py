from TinyFlow.Network import Network
from TinyFlow.Loss import Loss_CategoricalCrossEntropy
from TinyFlow.Optimizers import Optimizer_Adam, Optimizer_AdaGrad, Optimizer_RMSprop, Optimizer_SGD
from TinyFlow.Model import create_data

# Example dictionary which would be parsed from json
parsed = {
    "layers": [
        ["Dense", 64, 0, 1e-5, 0, 1e-5], ["ReLU", ], ["Dropout",
                                                      0.1], ["Dense", 128, 0, 0, 0, 0], ["ReLU", ], ["Dense", 3, 0, 0, 0, 0], ["Softmax", ]
    ],
    "loss": "CategoricalCrossEntropy",
    "optimizer": ["Adam", 0.05, 4e-8],
    "trainData": [1000, 3],
    "testData": [100, 3],
    "epoch": 10001,
}


def parse(parsed):
    # Create network
    net = Network(2)
    for layer in parsed["layers"]:
        if layer[0] is "Dense":
            net.addDenseLayer(layer[1], weight_regularizer_l1=layer[2], weight_regularizer_l2=layer[3],
                              bias_regulariser_l1=layer[4], bias_regulariser_l2=layer[5])
        elif layer[0] is "ReLU":
            net.addReLU()
        elif layer[0] is "Dropout":
            net.addDropoutLayer(layer[1])
        elif layer[0] is "Softmax":
            net.addSoftmax()
        else:
            print("------ ERROR PARSING LAYERS ------")
            raise ValueError('API faced issue while parsing layers')

    # Assigning loss
    if parsed["loss"] is "CategoricalCrossEntropy":
        lossFunction = Loss_CategoricalCrossEntropy()
    else:
        print("------ ERROR PARSING LOSS ------")
        raise ValueError('API faced issue while parsing loss')

    # Instantiating an optimizer
    if parsed["optimizer"][0] is "Adam":
        optimizer = Optimizer_Adam(
            learning_rate=parsed["optimizer"][1], decay=parsed["optimizer"][2])
    elif parsed["optimizer"][0] is "AdaGrad":
        optimizer = Optimizer_AdaGrad(
            learning_rate=parsed["optimizer"][1], decay=parsed["optimizer"][2])
    elif parsed["optimizer"][0] is "RMSprop":
        optimizer = Optimizer_RMSprop(
            learning_rate=parsed["optimizer"][1], decay=parsed["optimizer"][2])
    elif parsed["optimizer"][0] is "SGD":
        optimizer = Optimizer_SGD(
            learning_rate=parsed["optimizer"][1], decay=parsed["optimizer"][2])
    else:
        print("------ ERROR PARSING OPTIMIZER ------")
        raise ValueError('API faced issue while parsing optimizer')

    # Create Data
    X, y = create_data(parsed["trainData"][0], parsed["trainData"][1])

    # Start training
    net.train(X, y, parsed["epoch"], lossFunction, optimizer)

    # Test the accuracy on unseen data
    X_test, y_test = create_data(parsed["testData"][0], parsed["testData"][1])

    net.test(X_test, y_test, lossFunction)


if __name__ == "__main__":
    parse(parsed)
