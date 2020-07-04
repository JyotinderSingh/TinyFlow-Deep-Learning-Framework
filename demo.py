from TinyFlow.Layers import Layer_Dense, Layer_Dropout
from TinyFlow.Activations import Activation_ReLU, Activation_Softmax
from TinyFlow.Loss import Loss_CategoricalCrossEntropy
from TinyFlow.Model import create_data, model_accuracy
from TinyFlow.Optimizers import Optimizer_SGD, Optimizer_AdaGrad, Optimizer_RMSprop, Optimizer_Adam

# Create Dataset
# dimensions of the inputs is (1000, 2), the number if classes is 3
X, y = create_data(1000, 3)

# Create Dense layer with 2 input features and 64 output values
# first dense layer, 2 inputs (each sample has 2 featues), 64 outputs
dense1 = Layer_Dense(2, 512, weight_regularizer_l2=1e-5,
                     bias_regulariser_l2=1e-5)

# Create a ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

# Create a Dropout layer
dropout1 = Layer_Dropout(0.1)

# Create a second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(512, 3)  # second dense layer, 64 inputs, 3 outputs

# Create a Softmax activation (to be used with Dense layer)
activation2 = Activation_Softmax()

# Create a loss function
loss_function = Loss_CategoricalCrossEntropy()

# Create Optimizer
# optimizer = Optimizer_SGD(decay=1e-8, momentum=0.7)
# optimizer = Optimizer_AdaGrad(decay=1e-8)
# optimizer = Optimizer_RMSprop(learning_rate=0.05, decay=1e-8, rho=0.999)
optimizer = Optimizer_Adam(learning_rate=0.05, decay=4e-8)

# Train in loop
for epoch in range(10001):

    # Make a forward pass of our training data through this layer
    # the outputs are of the dimension (100, 3) [dot((100, 2), (2, 3))]
    dense1.forward(X)

    # Fwd pass through the actv function, takes output from previous layer
    # dimension of output from the ReLU is (100, 3)
    activation1.forward(dense1.output)

    # Make a forward pass through the dropout layer
    dropout1.forward(activation1.output)

    # Make a forward pass through second dense layer - takes the output
    # of the first activation function as the inputs
    # dimension of output from this layer is (100, 3)
    dense2.forward(dropout1.output)

    # Make a forward pass through the activation function - takes
    # the output of the previous layer
    activation2.forward(dense2.output)

    # Outputs of the first few samples
    # print(activation2.output[:5])

    # Calculate the loss from output of activation2 (Softmax activation)
    data_loss = loss_function.forward(activation2.output, y)
    regularization_loss = loss_function.regularization_loss(
        dense1) + loss_function.regularization_loss(dense2)

    loss = data_loss + regularization_loss

    # Print the loss value
    print('loss: ', loss)

    # Print the accuracy
    accuracy = model_accuracy(activation2.output, y)
    # print('acc: ', accuracy)

    if not epoch % 100:
        print(
            f'\nepoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, (data loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {optimizer.current_learning_rate}')

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dvalues)
    dense2.backward(activation2.dvalues)
    dropout1.backward(dense2.dvalues)
    activation1.backward(dropout1.dvalues)
    dense1.backward(activation1.dvalues)

    # Print gradients
    # print(dense1.dweights)
    # print(dense1.dbiases)
    # print(dense2.dweights)
    # print(dense2.dbiases)

    # Update weights
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


# Validate model

# Create TEST data set from the same distribution
X_test, y_test = create_data(100, 3)

# Make a forward pass of this test data through our Model
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Calculate the loss
loss = loss_function.forward(activation2.output, y_test)

# Calculate the accuracy from output of model and targets
accuracy = model_accuracy(activation2.output, y_test)

print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
