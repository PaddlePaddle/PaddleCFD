import logging

import paddle
from initializers import initializer_from_kwargs
from providers.path_provider import PathProvider
from utils.data_container import DataContainer
from utils.factory import create_collection
from utils.naming_util import snake_type_name


class ModelBase(paddle.nn.Layer):
    def __init__(
        self,
        input_shape=None,
        name=None,
        output_shape=None,
        ctor_kwargs=None,
        update_counter=None,
        path_provider: PathProvider = None,
        data_container: DataContainer = None,
        initializers=None,
        dynamic_ctx: dict = None,
        static_ctx: dict = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.update_counter = update_counter
        self.path_provider = path_provider
        self.data_container = data_container
        self._optim = None
        self.initializers = create_collection(
            initializers, initializer_from_kwargs, path_provider=self.path_provider
        )
        if dynamic_ctx is None:
            self.dynamic_ctx = {}
        else:
            self.dynamic_ctx = dynamic_ctx
        if static_ctx is None:
            self.static_ctx = {}
            if self.input_shape is not None:
                self.static_ctx["input_shape"] = tuple(self.input_shape)
        else:
            self.static_ctx = static_ctx
            if self.input_shape is None and "input_shape" in self.static_ctx:
                self.input_shape = self.static_ctx["input_shape"]
        self.name = name or snake_type_name(self)
        self.ctor_kwargs = ctor_kwargs
        if self.ctor_kwargs is not None and "update_counter" in self.ctor_kwargs:
            self.ctor_kwargs.pop("update_counter")
        self.is_initialized = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def submodels(self):
        raise NotImplementedError

    def clear_buffers(self):
        raise NotImplementedError

    @property
    def is_batch_size_dependent(self):
        raise NotImplementedError

    def initialize(self, lr_scale_factor=None):
        self.initialize_weights()
        self.initialize_optim(lr_scale_factor=lr_scale_factor)
        self.apply_initializers()
        self.is_initialized = True
        return self

    def initialize_weights(self):
        raise NotImplementedError

    def apply_initializers(self):
        raise NotImplementedError

    def initialize_optim(self, lr_scale_factor=None):
        raise NotImplementedError

    def model_specific_initialization(self):
        pass

    @property
    def optim(self):
        return self._optim

    def optim_step(self, grad_scaler):
        raise NotImplementedError

    def optim_schedule_step(self):
        raise NotImplementedError

    def optim_zero_grad(self, set_to_none=True):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError

    def before_accumulation_step(self):
        """before_accumulation_step hook (e.g. for freezers)"""
        for model in self.submodels.values():
            model.before_accumulation_step()

    def after_update_step(self):
        """after_update_step hook (e.g. for EMA)"""
        pass
