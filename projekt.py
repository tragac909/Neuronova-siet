import numpy as np
from time import time
from utils import *

class MultipleHiddenLayerNetwork:
    def __init__(self,
                 input_dim,
                 hidden_layers,  # kolko neuronov ma mat skryty layer? [60, 50]
                 output_dim, # 1
                 activations,  # aku aktivacnu funkciu ma layer pouzivat? ["tanh", "tanh"]
                 biases=None,  # pridat danemu layeru bias? [True, False]
                 weight_init="random",  # e.g. "random"
                 learning_rate=0.01, 
                 momentum=0.0, # defaultne ziadne momentum
                 lr_schedule=None,
                 loss_function="mse",  # "mse", "logcosh", or "huber"
                 huber_delta=1.0, # parameter pre Huber loss
                 lr_decay_rate=1.0): # alpha sa updatuje kazdych 100 epoch

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activations = activations
        self.loss_function = loss_function
        self.huber_delta = huber_delta
        self.lr_decay_rate = lr_decay_rate

        self.base_lr = learning_rate

        if biases is None:
            biases = [True] * (len(hidden_layers) + 1)
        self.biases = biases

        self.weight_init = weight_init
        self.alpha = learning_rate
        self.momentum = momentum
        self.lr_schedule = lr_schedule

        if len(self.hidden_layers) != len(self.activations) - 1:
            raise ValueError("Pocet skrytych vrstiev musi by rovny len(activations) - 1.")
        if len(self.biases) != len(hidden_layers) + 1:
            raise ValueError("Pocet biasov musi byt rovny num_hidden_layers + 1.")

        self.weights, self.velocities = self.initialize_weights_and_velocities()

    def initialize_weights_and_velocities(self):
        weights = []
        velocities = []
        layer_dims = [self.input_dim] + self.hidden_layers + [self.output_dim]

        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i] + (1 if self.biases[i] else 0)
            out_dim = layer_dims[i + 1]

            if self.weight_init == "random":
                W = np.random.randn(out_dim, in_dim)
            elif self.weight_init == "xavier":
                W = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / (layer_dims[i] + layer_dims[i+1]))
            elif self.weight_init == "he":
                W = np.random.randn(out_dim, in_dim) * np.sqrt(2.0 / layer_dims[i])
            elif self.weight_init == "lecun":
                W = np.random.randn(out_dim, in_dim) * np.sqrt(1.0 / layer_dims[i])
            else:
                raise ValueError("Unsupported weight initialization method.")

            weights.append(W)
            velocities.append(np.zeros_like(W))

        return weights, velocities

    def add_bias(self, x):
        """
        Appends 1.0 if x is 1D.  e.g. x shape (in_dim,) => (in_dim+1,)
        """
        return np.concatenate([x, [1.0]])

    def forward(self, x):
        nets = []
        activations = []

        if self.biases[0]:
            a = self.add_bias(x)
        else:
            a = x

        for i, W in enumerate(self.weights):
            net = W @ a
            nets.append(net)

            act_fn = self.activations[i]
            if act_fn == "tanh":
                a = self.tanh(net)
            elif act_fn == "sigmoid":
                a = self.sigmoid(net)
            elif act_fn == "relu":
                a = self.relu(net)
            elif act_fn == "linear":
                a = net
            else:
                raise ValueError("Unsupported activation function")

            if i < len(self.weights) - 1 and self.biases[i + 1]:
                a = self.add_bias(a)

            activations.append(a)

        return activations

    def backward(self, x, d,  activations):
        """
        Single-sample backprop with momentum, given forward pass results.

        x: shape (input_dim,)
        d: shape (output_dim,)
        nets, activations are lists from forward(x).
        """
        num_layers = len(self.weights)
        deltas = [None] * num_layers

        # Output layer delta
        out_idx = num_layers - 1
        y_hat = activations[out_idx]  # shape = (output_dim,) if no bias appended
        error = d - y_hat

        # Activation derivative of the output
        def act_deriv(act_name, val):
            if act_name == "tanh":
                return self.tanh_derivative(val)
            elif act_name == "sigmoid":
                return self.sigmoid_derivative(val)
            elif act_name == "relu":
                return self.relu_derivative(val)
            elif act_name == "linear":
                return np.ones_like(val)
            else:
                raise ValueError("Unsupported activation.")

        out_deriv = act_deriv(self.activations[out_idx], y_hat)
        deltas[out_idx] = error * out_deriv

        for i in reversed(range(num_layers - 1)):
            W_next = self.weights[i + 1]
            delta_next = deltas[i + 1]

            # If the next layer had a bias, skip the last column
            if self.biases[i + 1]:
                W_for_delta = W_next[:, :-1]
            else:
                W_for_delta = W_next

            # raw_delta_i
            raw_delta_i = W_for_delta.T @ delta_next

            # Activation derivative for this layer
            a_i = activations[i]
            # If this layer appended a bias (i < num_layers-1 and biases[i+1]), remove it
            if i < num_layers - 1 and self.biases[i + 1]:
                a_i = a_i[:-1]  # shape minus bias

            deriv_i = act_deriv(self.activations[i], a_i)
            deltas[i] = raw_delta_i * deriv_i

        # Weight updates with momentum
        for i in range(num_layers):
            # input to layer i
            if i == 0:
                # for first layer, it's x plus bias if biases[0]
                if self.biases[0]:
                    prev_a = self.add_bias(x)
                else:
                    prev_a = x
            else:
                prev_a = activations[i - 1]

            grad_i = np.outer(deltas[i], prev_a)

            self.velocities[i] = self.momentum * self.velocities[i] + self.alpha * grad_i
            self.weights[i] += self.velocities[i]

    def save_weights(model, file_name):
        """
        Ulozi model do suboru, v prvej polovici su jeho parametre a v druhej vahy.
        """
        with open(file_name, 'w') as f:
            # Write metadata header
            f.write("# Model Metadata\n")
            f.write(f"input_dim: {model.input_dim}\n")
            f.write(f"hidden_layers: {model.hidden_layers}\n")
            f.write(f"output_dim: {model.output_dim}\n")
            f.write(f"activations: {model.activations}\n")
            f.write(f"biases: {model.biases}\n")
            f.write(f"weight_init: {model.weight_init}\n")
            f.write(f"learning_rate: {model.alpha}\n")
            f.write(f"momentum: {model.momentum}\n")
            f.write(f"loss_function: {model.loss_function}\n")
            f.write(f"huber_delta: {model.huber_delta}\n")
            f.write(f"lr_decay_rate: {model.lr_decay_rate}\n")
            f.write("\n# Weights\n")

            # Save each layer's weights as a 2D array
            for i, W in enumerate(model.weights):
                f.write(f"# Layer {i} weights: shape {W.shape}\n")
                np.savetxt(f, W)  # Ensure weights are saved as 2D arrays
                f.write("\n")

    def huber_loss(self, y, y_hat):
        # ako mse, ale menej citliva na outlayery
        delta = self.huber_delta
        diff = y - y_hat
        abs_diff = np.abs(diff)

        quadratic = 0.5 * (diff ** 2)
        linear = delta * (abs_diff - 0.5 * delta)
        return np.sum(np.where(abs_diff <= delta, quadratic, linear))

    def mse_loss(self, y, y_hat):
        return np.sum((y - y_hat) ** 2)

    def compute_loss(self, y, y_hat):
        if self.loss_function == "mse":
            return self.mse_loss(y, y_hat)
        elif self.loss_function == "huber":
            return self.huber_loss(y, y_hat)
        else:
            raise ValueError("Unsupported loss function")

    def change_alpha(self, epoch):
        # zavola sa sice vzdy, ale iba kazdu 100-tu epochu zmeni learning rate kvoli stabilite
        if (epoch + 1) % 100 == 0:
            self.alpha *= self.lr_decay_rate
            print("new learning rate= ", self.alpha)

    def train(self, X, D, epochs=1000):
        n_samples = X.shape[1]
        loss_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0

            self.change_alpha(epoch)
            for i in indices:
                x = X[:, i]
                d = D[:, i]

                activations = self.forward(x)

                y_hat = activations[-1]
                total_loss += self.compute_loss(d, y_hat)
                self.backward(x, d,  activations)

            loss_history.append(total_loss)

            # vypis
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss = {total_loss:.4f}")

        # show_history() z utils.py vyzaduje takyto tvar
        return {"loss": loss_history}

    def predict(self, X):
        n_samples = X.shape[1]
        Y_hat = np.zeros((self.output_dim, n_samples))

        for i in range(n_samples):
            acts = self.forward(X[:, i])
            Y_hat[:, i] = acts[-1]

        return Y_hat

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(y):
        return 1.0 - y ** 2

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(y):
        return y * (1.0 - y)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(y):
        return (y > 0).astype(float)


def prepare_data(file_path):
    """ Nacita data zo suboru """
    data = np.loadtxt(file_path)
    X = data[:, :2].T
    D = data[:, 2][np.newaxis, :]
    return X, D


def split_data(X, D, test_size=0.2):
    """ Najprv spravi permutaciu, az potom splitne a vrati"""
    n_samples = X.shape[1]
    indices = np.random.permutation(n_samples)
    split_point = int(n_samples * (1 - test_size))

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    X_train = X[:, train_indices]
    D_train = D[:, train_indices]

    X_test = X[:, test_indices]
    D_test = D[:, test_indices]

    return X_train, D_train, X_test, D_test


def load_weights(file_name):
    """
    Nacita metadata a vahy zo suboru a vrati model
    Returns:
        instanciu MultipleHiddenLayerNetwork s parametrami a vahami zo suboru.
    """
    with open(file_name, 'r') as f:
        content = f.read()

    # Split the file into metadata and weights
    parts = content.split("\n# Weights\n")
    metadata_block, weights_block = parts[0], parts[1]

    # Parse metadata
    metadata = {}
    for line in metadata_block.splitlines():
        if line.startswith("#") or not line.strip():
            continue  # Skip comments and blank lines
        key, value = line.split(": ", 1)
        if key == "hidden_layers":
            metadata[key] = [int(x) for x in value.strip("[]").split(",")]
        elif key == "activations" or key == "biases":
            metadata[key] = eval(value)  # Parse list-like strings
        elif key in {"input_dim", "output_dim"}:
            metadata[key] = int(value)
        elif key in {"learning_rate", "momentum", "huber_delta, lr_decay_rate"}:
            metadata[key] = float(value)
        else:
            metadata[key] = value.strip()

    # Parse weights
    weight_blocks = weights_block.strip().split("\n\n")
    weights = []
    expected_dims = [metadata["input_dim"]] + metadata["hidden_layers"] + [metadata["output_dim"]]
    biases = metadata["biases"]

    for i, (block, in_dim, out_dim) in enumerate(zip(weight_blocks, expected_dims[:-1], expected_dims[1:])):
        weight_array = np.loadtxt(block.splitlines())

        # Ensure weight_array is 2D and matches expected shape
        expected_shape = (out_dim, in_dim + (1 if biases[i] else 0))
        if weight_array.shape != expected_shape:
            # Attempt to reshape if it's a 1D array
            if weight_array.ndim == 1 and weight_array.size == np.prod(expected_shape):
                weight_array = weight_array.reshape(expected_shape)
            else:
                raise ValueError(f"Layer {i} weight shape mismatch: expected {expected_shape}, got {weight_array.shape}")

        weights.append(weight_array)

    # Create a new model
    model = MultipleHiddenLayerNetwork(
        input_dim=metadata["input_dim"],
        hidden_layers=metadata["hidden_layers"],
        output_dim=metadata["output_dim"],
        activations=metadata["activations"],
        biases=metadata["biases"],
        weight_init=metadata["weight_init"],
        learning_rate=metadata["learning_rate"],
        momentum=metadata["momentum"],
        loss_function=metadata["loss_function"],
        huber_delta=metadata["huber_delta"],
        lr_decay_rate=metadata["lr_decay_rate"]
    )

    # Assign weights to the model
    model.weights = weights
    return model


def evaluate(file_path, load_path=''):
    """
    Ak je load False:
        nacita data zo suboru a natrenuje z nich model
    Ak je load True:
        nacita natrenovane vahy zo suboru a spravi z nich model
    Potom predikuje a evaluuje.
    Returns:
         model a pocet bodov z kola
    """
    all_inputs, all_targets = prepare_data(file_path)
    X_train, D_train, X_test, D_test = split_data(all_inputs, all_targets)
    input_dim = X_train.shape[0]
    output_dim = D_train.shape[0]

    #######################################################
    # Vase test data (ak je mate cely subor z testovacich dat):
    # X_test, D_test = prepare_data(vasa_file_path)
    #######################################################

    if load_path != '': # nacitavame vahy zo suboru
        model = load_weights(load_path)
    else:
        model = MultipleHiddenLayerNetwork(
            input_dim=input_dim,
            hidden_layers=[55, 35, 20],
            output_dim=output_dim,
            activations=["tanh", "sigmoid", "sigmoid", "linear"],
            biases=[True, True, True, True],
            weight_init="random",
            learning_rate=0.007,
            momentum=0.7,
            loss_function="mse",
            huber_delta=1.5,
            lr_decay_rate=0.985 # pre 2000 epoch
        )
        start = time()
        history = model.train(X_train, D_train, epochs=500)
        duration = time() - start
        print(f"\nTraining duration: {int(duration // 60)} minutes and {int(duration % 60)} seconds")


    Y_hat = model.predict(X_test)
    mse = np.mean((D_test - Y_hat) ** 2)
    print(f"Final MSE on test data: {mse:.4f}")
    Y_hat_all = model.predict(all_inputs)
    mse_all = np.mean((all_targets - Y_hat_all) ** 2)
    print(f"Final MSE on all data: {mse_all:.4f}")
    body = (16 - 75*mse) * 0.7  +  (16 - 75*mse_all) * 0.3

    ###########################################
    # Kreslenie grafov
    show_data(X_test, D_test, predicted=Y_hat)
    if not load_path != '': # history sa neuklada do ulozeneho modelu
        show_history(history)
    ###########################################
    return model, body



if __name__ == "__main__":
    file_path = 'mlp_train.txt'
    # model sa sam natrenuje z dat a ulozi do noveho suboru
    print("Manualne natrenovanie")
    model, prve_kolo = evaluate(file_path)
    model.save_weights('model_weights.txt')

    # nacita a evaluuje model z pretrenovanej siete zo suboru
    print('\nUlozeny predtrenovany model (1500 epoch):')
    pretrained_path = 'best.txt'
    pretrained_model, druhe_kolo = evaluate(file_path, pretrained_path)

    print("Orientacne body = ", prve_kolo * 0.7  +  druhe_kolo * 0.3)
