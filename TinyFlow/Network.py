# Network class
class Network:

    def __init__(self):

        # Create a list of all Network objects
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def compile_model(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers
        self.trainable_layers = []

        # Iterate through all the objects
        for i in range(layer_count):

            # If this is the first layer, the input layer will be considered
            # as the previous object
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # All layers except first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # The last layer will have the next object as the loss
            # We also save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If the layer contains an attribute called 'weights', it's a
            # trainable layer, add it to the list of trainable layers
            # We dont't need to check for biases, checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            # Update the loss object with trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)

    def forward(self, X, training):

        # Call the forward method on the input layer
        # This will set the output property that the first layer
        # is expecting as the 'prev' object
        self.input_layer.forward(X, training)

        # Call the forward methid of every object in sequence
        # while passing the output property of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # 'layer' is now the last object from the list, return its output
        return layer.output

    def backward(self, output, y):

        # First call the backward method on loss
        # This will set the dvalues property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects in
        # reversed order, passing down the dvalues as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dvalues)

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1, epochs+1):

            # Perform the forward pass
            output = self.forward(X, training=True)

            # Calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)

            loss = data_loss + regularization_loss

            # Get predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'\nepoch: {epoch}\nacc: {accuracy:.3f}, loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate}')

        # If validation data has been provided
        if validation_data is not None:

            # For better readability
            X_val, y_val = validation_data

            # Perform forward pass
            output = self.forward(X_val, training=False)

            # Calculate the loss
            loss = self.loss.calculate(output, y_val)

            # Get predictions and calculate validation accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Print a sumarry
            print(f'\nvalidation,\nacc: {accuracy:.3f}, loss: {loss:.3f}')

class Layer_Input:

    # Pass the input
    def forward(self, inputs, training):
        self.output = inputs
