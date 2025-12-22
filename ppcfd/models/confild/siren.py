import math
from collections import OrderedDict

import numpy as np
import paddle


DEFAULT_W0 = 30.0


###################### ConFILD Model #######################
class Swish(paddle.nn.Layer):
    """
    Swish activation function: f(x) = x * sigmoid(x).

    A smooth, non-monotonic activation function that has been shown to work
    better than ReLU on deeper models across a number of challenging datasets.
    """

    def __init__(self):
        super().__init__()
        self.Sigmoid = paddle.nn.Sigmoid()

    def forward(self, x):
        """
        Apply Swish activation.

        Args:
            x (paddle.Tensor): Input tensor.

        Returns:
            paddle.Tensor: Output tensor with same shape as input.
        """
        return x * self.Sigmoid(x)


class Sine(paddle.nn.Layer):
    """
    Sine activation function for SIREN (Sinusoidal Representation Networks).

    Args:
        w0 (float, optional): Frequency parameter for sine activation. Defaults to DEFAULT_W0 (30.0).
    """

    def __init__(self, w0=DEFAULT_W0):
        self.w0 = w0
        super().__init__()

    def forward(self, input):
        """
        Apply sine activation with frequency modulation.

        Args:
            input (paddle.Tensor): Input tensor.

        Returns:
            paddle.Tensor: sin(w0 * input).
        """
        return paddle.sin(self.w0 * input)


def sine_init(m, w0=DEFAULT_W0):
    """
    Weight initialization for SIREN hidden layers.

    Initializes weights uniformly in [-√(6/n)/w0, √(6/n)/w0] where n is input dimension.
    This initialization is critical for maintaining stable signal propagation in SIREN networks.

    Args:
        m (paddle.nn.Layer): Layer to initialize (must have 'weight' attribute).
        w0 (float, optional): Frequency parameter. Defaults to DEFAULT_W0.
    """
    with paddle.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.shape[-1]
            m.weight.uniform_(min=-math.sqrt(6 / num_input) / w0, max=math.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    """
    Weight initialization for SIREN first layer.

    Initializes weights uniformly in [-1/n, 1/n] where n is input dimension.
    Different from hidden layers to handle raw coordinate inputs properly.

    Args:
        m (paddle.nn.Layer): Layer to initialize (must have 'weight' attribute).
    """
    with paddle.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.shape[-1]
            m.weight.uniform_(min=-1 / num_input, max=1 / num_input)


def __check_Linear_weight(m):
    if isinstance(m, paddle.nn.Linear):
        if hasattr(m, "weight"):
            return True
    return False


def init_weights_normal(m):
    if __check_Linear_weight(m):
        init_KaimingNormal = paddle.nn.initializer.KaimingNormal(nonlinearity="relu", negative_slope=0.0)
        init_KaimingNormal(m.weight)


def init_weights_selu(m):
    if __check_Linear_weight(m):
        num_input = m.weight.shape[-1]
        init_Normal = paddle.nn.initializer.Normal(std=1 / math.sqrt(num_input))
        init_Normal(m.weight)


def init_weights_elu(m):
    if __check_Linear_weight(m):
        num_input = m.weight.shape[-1]
        init_Normal = paddle.nn.initializer.Normal(std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))
        init_Normal(m.weight)


def init_weights_xavier(m):
    if __check_Linear_weight(m):
        init_XavierNormal = paddle.nn.initializer.XavierNormal()
        init_XavierNormal(m.weight)


NLS_AND_INITS = {
    "sine": (Sine(), sine_init, first_layer_sine_init),
    "relu": (paddle.nn.ReLU(), init_weights_normal, None),
    "sigmoid": (paddle.nn.Sigmoid(), init_weights_xavier, None),
    "tanh": (paddle.nn.Tanh(), init_weights_xavier, None),
    "selu": (paddle.nn.SELU(), init_weights_selu, None),
    "softplus": (paddle.nn.Softplus(), init_weights_normal, None),
    "elu": (paddle.nn.ELU(), init_weights_elu, None),
    "swish": (Swish(), init_weights_xavier, None),
}


class BatchLinear(paddle.nn.Linear):
    """
    Batch-wise linear layer with external parameter injection support.

    Extends paddle.nn.Linear to support explicit parameter passing,
    useful for meta-learning and hypernetwork applications.

    Automatically handles Paddle version differences in weight layout:
    - Paddle 2.x: weight.shape = (out_features, in_features)
    - Paddle 3.x: weight.shape = (in_features, out_features)

    Args:
        in_features (int): Size of input features.
        out_features (int): Size of output features.
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        # Auto-detect Paddle version by checking weight layout
        self._weight_is_transposed = self.weight.shape[0] == out_features

    def forward(self, input, params=None):
        """
        Forward pass with optional external parameters.

        Args:
            input (paddle.Tensor): Input tensor of shape (..., in_features).
            params (OrderedDict, optional): External parameters dict. Defaults to None.

        Returns:
            paddle.Tensor: Output tensor of shape (..., out_features).
        """
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get("bias", None)
        weight = params["weight"]

        # Transpose weight if needed (Paddle 2.x compatibility)
        if self._weight_is_transposed:
            weight = paddle.transpose(weight, perm=[1, 0])

        output = paddle.matmul(input, weight)

        if bias is not None:
            output += bias.unsqueeze(axis=-2)

        return output


class FeatureMapping:
    """
    Feature mapping class for Fourier Feature Networks.

    Supports multiple mapping strategies including Gaussian random Fourier features,
    positional encoding, and radial basis functions (RBF) for improving coordinate-based
    neural network representations.

    Reference:
        Tancik et al. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    """

    def __init__(
        self,
        in_features,
        mode="basic",
        gaussian_mapping_size=256,
        gaussian_rand_key=0,
        gaussian_tau=1.0,
        pe_num_freqs=4,
        pe_scale=2,
        pe_init_scale=1,
        pe_use_nyquist=True,
        pe_lowest_dim=None,
        rbf_out_features=None,
        rbf_range=1.0,
        rbf_std=0.5,
    ):
        """
        Initialize feature mapping.

        Args:
            in_features (int): Number of input features.
            mode (str, optional): Mapping mode. Options: "basic", "gaussian", "positional", "rbf". Defaults to "basic".
            gaussian_mapping_size (int, optional): Output dimension for Gaussian mapping. Defaults to 256.
            gaussian_rand_key (int, optional): Random seed for Gaussian mapping. Defaults to 0.
            gaussian_tau (float, optional): Standard deviation for Gaussian mapping. Defaults to 1.0.
            pe_num_freqs (int, optional): Number of frequency bands for positional encoding. Defaults to 4.
            pe_scale (int, optional): Base scale for frequencies in positional encoding. Defaults to 2.
            pe_init_scale (int, optional): Initial scale multiplier for positional encoding. Defaults to 1.
            pe_use_nyquist (bool, optional): Use Nyquist frequency to determine num_freqs. Defaults to True.
            pe_lowest_dim (int, optional): Lowest dimension for Nyquist calculation. Defaults to None.
            rbf_out_features (int, optional): Number of RBF centers. Defaults to None.
            rbf_range (float, optional): Range for RBF center initialization. Defaults to 1.0.
            rbf_std (float, optional): Standard deviation for RBF kernels. Defaults to 0.5.
        """
        self.mode = mode
        if mode == "basic":
            self.B = np.eye(in_features)
        elif mode == "gaussian":
            rng = np.random.default_rng(gaussian_rand_key)
            self.B = rng.normal(loc=0.0, scale=gaussian_tau, size=(gaussian_mapping_size, in_features))
        elif mode == "positional":
            if pe_use_nyquist and pe_lowest_dim:
                pe_num_freqs = self.get_num_frequencies_nyquist(pe_lowest_dim)
            self.B = pe_init_scale * np.vstack([(pe_scale**i * np.eye(in_features)) for i in range(pe_num_freqs)])
            self.dim = tuple(self.B.shape)[0] * 2
        elif mode == "rbf":
            self.centers = paddle.nn.Parameter(paddle.empty(shape=(rbf_out_features, in_features), dtype="float32"))
            self.sigmas = paddle.nn.Parameter(paddle.empty(shape=rbf_out_features, dtype="float32"))
            init_Uniform = paddle.nn.initializer.Uniform(low=-1 * rbf_range, high=rbf_range)
            init_Uniform(self.centers)
            init_Constant = paddle.nn.initializer.Constant(value=rbf_std)
            init_Constant(self.sigmas)

    def __call__(self, input):
        if self.mode in ["basic", "gaussian", "positional"]:
            return self.fourier_mapping(input, self.B)
        elif self.mode == "rbf":
            return self.rbf_mapping(input)

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    @staticmethod
    def fourier_mapping(x, B):
        """
        Apply Fourier feature mapping: [sin(2πxB^T), cos(2πxB^T)].

        Args:
            x (paddle.Tensor): Input coordinates of shape (..., in_features).
            B (np.ndarray): Frequency matrix of shape (mapping_size, in_features).

        Returns:
            paddle.Tensor: Fourier features of shape (..., 2 * mapping_size).
        """
        if B is None:
            return x
        else:
            B = paddle.to_tensor(data=B, dtype="float32", place=x.place)
            x_proj = 2.0 * np.pi * x @ B.T
            return paddle.concat([paddle.sin(x_proj), paddle.cos(x_proj)], axis=-1)

    def rbf_mapping(self, x):
        size = tuple(x.shape)[:-1] + tuple(self.centers.shape)
        x = x.unsqueeze(axis=-2).expand(shape=size)
        distances = paddle.pow(x - self.centers, 2).sum(axis=-1) * self.sigmas
        return self.gaussian(distances)

    @staticmethod
    def gaussian(alpha):
        phi = paddle.exp(-1 * paddle.pow(alpha, 2))
        return phi


class SIRENAutodecoder_film(paddle.nn.Layer):
    """
    SIREN (Sinusoidal Representation Networks) with FiLM conditioning for autodecoding.

    This architecture uses sine activations and latent code modulation (FiLM) for
    implicit neural representations. It takes both coordinate inputs and latent codes,
    making it suitable for learning multiple shapes/scenes with a single network.

    Reference:
        Sitzmann et al. "Implicit Neural Representations with Periodic Activation Functions" (NeurIPS 2020)

    Args:
        input_keys (Tuple[str, ...], optional): Keys to get input tensors from dict. First key for coordinates, second for latents.
        output_keys (Tuple[str, ...], optional): Keys to save output tensors into dict.
        in_coord_features (int, optional): Number of input coordinate features (e.g., 2 for 2D, 3 for 3D).
        in_latent_features (int, optional): Number of latent features for conditioning.
        out_features (int, optional): Number of output features (e.g., 3 for RGB).
        num_hidden_layers (int, optional): Number of hidden layers.
        hidden_features (int, optional): Number of hidden layer features.
        outermost_linear (bool, optional): Whether to use linear layer at output. Defaults to False.
        nonlinearity (str, optional): Activation function. Options: "sine", "relu", "tanh", etc. Defaults to "sine".
        weight_init (Callable, optional): Custom weight initialization function. Defaults to None.
        bias_init (Callable, optional): Custom bias initialization function. Defaults to None.
        premap_mode (str, optional): Feature mapping mode before network. Options: "gaussian", "positional", "rbf". Defaults to None.

    Examples:
        >>> import ppsci
        >>> model = ppsci.arch.SIRENAutodecoder_film(
        ...     input_keys=["coords", "latents"],
        ...     output_keys=("output",),
        ...     in_coord_features=2,
        ...     in_latent_features=128,
        ...     out_features=3,
        ...     num_hidden_layers=10,
        ...     hidden_features=128,
        ... )
        >>> input_data = {
        ...     "coords": paddle.randn([1000, 2]),
        ...     "latents": paddle.randn([1000, 128])
        ... }
        >>> out_dict = model(input_data)
        >>> print(out_dict["output"].shape)
        [1000, 3]
    """

    def __init__(
        self,
        input_keys,
        output_keys,
        in_coord_features,
        in_latent_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        outermost_linear=False,
        nonlinearity="sine",
        weight_init=None,
        bias_init=None,
        premap_mode=None,
        **kwargs,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.premap_mode = premap_mode
        if self.premap_mode is not None:
            self.premap_layer = FeatureMapping(in_coord_features, mode=premap_mode, **kwargs)
            in_coord_features = self.premap_layer.dim
        self.first_layer_init = None
        self.nl, nl_weight_init, first_layer_init = NLS_AND_INITS[nonlinearity]
        if weight_init is not None:
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net1 = paddle.nn.LayerList(
            sublayers=[BatchLinear(in_coord_features, hidden_features)]
            + [BatchLinear(hidden_features, hidden_features) for i in range(num_hidden_layers)]
            + [BatchLinear(hidden_features, out_features)]
        )

        self.net2 = paddle.nn.LayerList(
            sublayers=[
                BatchLinear(in_latent_features, hidden_features, bias_attr=False) for i in range(num_hidden_layers + 1)
            ]
        )

        if self.weight_init is not None:
            self.net1.apply(self.weight_init)
            self.net2.apply(self.weight_init)
        if first_layer_init is not None:
            self.net1[0].apply(first_layer_init)
            self.net2[0].apply(first_layer_init)
        if bias_init is not None:
            self.net2.apply(bias_init)

    def forward(self, input_data):
        coords = input_data[self.input_keys[0]]
        latents = input_data[self.input_keys[1]]

        if self.premap_mode is not None:
            x = self.premap_layer(coords)
        else:
            x = coords

        for i in range(len(self.net1) - 1):
            x = self.net1[i](x) + self.net2[i](latents)
            x = self.nl(x)

        x = self.net1[-1](x)

        return {self.output_keys[0]: x}

    def disable_gradient(self):
        for param in self.parameters():
            param.stop_gradient = True


class LatentContainer(paddle.nn.Layer):
    """
    Learnable latent code container for autodecoding applications.

    This module stores and retrieves per-sample latent codes, which can be used
    for representing multiple instances (shapes, scenes) with a single decoder network.
    Supports multi-GPU training and different dimensional arrangements.

    Reference:
        Park et al. "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation" (CVPR 2019)

    Args:
        input_keys (Tuple[str, ...], optional): Key to get batch indices from dict. Defaults to ("input",).
        output_keys (Tuple[str, ...], optional): Key to save latent codes into dict. Defaults to ("output",).
        N_samples (int, optional): Total number of samples/instances in dataset. Defaults to None.
        N_features (int, optional): Dimension of latent codes. Defaults to None.
        dims (int, optional): Number of spatial dimensions (for proper broadcasting). Defaults to None.
        lumped (bool, optional): If True, adds single dimension; if False, adds dims dimensions. Defaults to False.

    Examples:
        >>> import ppsci
        >>> import paddle
        >>> model = ppsci.arch.LatentContainer(
        ...     N_samples=1600,
        ...     N_features=128,
        ...     dims=2,
        ...     lumped=True
        ... )
        >>> batch_indices = paddle.randint(0, 1600, [32], dtype='int64')
        >>> input_dict = {"input": batch_indices}
        >>> out_dict = model(input_dict)
        >>> print(out_dict["output"].shape)
        [32, 1, 128]
    """

    def __init__(
        self,
        input_keys=("input",),
        output_keys=("output",),
        N_samples=None,
        N_features=None,
        dims=None,
        lumped=False,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.dims = [1] * dims if not lumped else [1]
        self.expand_dims = " ".join(["1" for _ in range(dims)]) if not lumped else "1"
        self.expand_dims = f"N f -> N {self.expand_dims} f"
        self.latents = self.create_parameter(
            shape=(N_samples, N_features),
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

    def forward(self, batch_ids):
        x = batch_ids[self.input_keys[0]]
        selected_latents = paddle.gather(self.latents, x)
        if len(selected_latents.shape) > 1:
            getShape = [tuple(selected_latents.shape)[0]] + self.dims + [tuple(selected_latents.shape)[1]]
        else:
            getShape = [-1] + self.dims
        expanded_latents = selected_latents.reshape(getShape)
        return {self.output_keys[0]: expanded_latents}
