from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class RansBaselineModel(CompositeModelBase):
    def __init__(self, latent, decoder, encoder=None, **kwargs):
        super().__init__(**kwargs)
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        self.encoder = create(
            encoder, model_from_kwargs, input_shape=self.input_shape, **common_kwargs
        )
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            **common_kwargs
        )
        self.decoder = create(
            decoder,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.latent.output_shape,
            output_shape=self.output_shape
        )

    @property
    def submodels(self):
        return dict(
            **dict(encoder=self.encoder) if self.encoder is not None else {},
            latent=self.latent,
            decoder=self.decoder
        )

    def forward(
        self, mesh_pos, grid_pos, query_pos, mesh_to_grid_edges, grid_to_query_edges
    ):
        outputs = {}
        encoded = self.encoder(
            mesh_pos=mesh_pos, grid_pos=grid_pos, mesh_to_grid_edges=mesh_to_grid_edges
        )
        propagated = self.latent(encoded)
        x_hat = self.decoder(
            propagated, query_pos=query_pos, grid_to_query_edges=grid_to_query_edges
        )
        outputs["x_hat"] = x_hat
        return outputs
