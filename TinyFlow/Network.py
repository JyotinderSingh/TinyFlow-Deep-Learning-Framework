######################
## UNDER DEVELOPMENT##
######################

# Network class
class Network:

    def __init__(self):

        # Create a list of all Network objects
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def compile_model(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

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
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss

    def forward(self, X):

        # Call the forward method on the input layer
        # This will set the output property that the first layer
        # is expecting as the 'prev' object
        self.input_layer.forward(X)

        # Call the forward methid of every object in sequence
        # while passing the output property of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output)

        # 'layer' is now the last object from the list, return its output
        return layer.output

    def train(self, X, y, *, epochs=1, print_every=1):

        # Main training loop
        for epoch in range(1, epochs+1):

            # Perform the forward pass
            output = self.forward(X)

            # Temporary
            print(output)
            exit()




class Layer_Input:

    # Pass the input
    def forward(self, inputs):
        self.output = inputs
