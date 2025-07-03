import copy
import sys

import paddle
import paddle.nn as nn

from ..neuralop.models import FNO
from .base_model import BaseModel
from .integral_finetuning import Integral_Cd
from .neighbor_ops import NeighborMLPConvLayer
from .neighbor_ops import NeighborMLPConvLayerLinear
from .neighbor_ops import NeighborMLPConvLayerWeighted
from .neighbor_ops import NeighborSearchLayer
from .net_utils import MLP
from .net_utils import AdaIN
from .net_utils import PositionalEmbedding
from .net_utils import Projection
from .utilities3 import count_params
from .utilities3 import memory_usage
from .utilities3 import num_of_nans
from .utilities3 import paddle_memory_usage
from .utilities3 import show_tensor_range


class GNOFNOGNO(BaseModel):
    def __init__(
        self,
        radius_in=0.05,
        radius_out=0.05,
        embed_dim=64,
        hidden_channels=(32, 32),
        in_channels=1,
        out_channels=1,
        fno_modes=(16, 16, 16),
        fno_hidden_channels=32,
        fno_out_channels=32,
        fno_domain_padding=0.125,
        fno_norm="group_norm",
        fno_factorization="tucker",
        fno_rank=0.4,
        linear_kernel=True,
        weighted_kernel=True,
    ):
        super().__init__()
        self.weighted_kernel = weighted_kernel
        self.nb_search_in = NeighborSearchLayer(radius_in)
        self.nb_search_out = NeighborSearchLayer(radius_out)
        self.pos_embed = PositionalEmbedding(embed_dim)
        self.df_embed = MLP([in_channels, embed_dim, 3 * embed_dim], paddle.nn.GELU)
        self.linear_kernel = linear_kernel
        kernel1 = MLP([10 * embed_dim, 512, 256, hidden_channels[0]], paddle.nn.GELU)
        self.gno1 = NeighborMLPConvLayerWeighted(mlp=kernel1)
        if linear_kernel == False:
            kernel2 = MLP(
                [fno_out_channels + 4 * embed_dim, 512, 256, hidden_channels[1]],
                paddle.nn.GELU,
            )
            self.gno2 = NeighborMLPConvLayer(mlp=kernel2)
        else:
            kernel2 = MLP([7 * embed_dim, 512, 256, hidden_channels[1]], paddle.nn.GELU)
            self.gno2 = NeighborMLPConvLayerLinear(mlp=kernel2)
        self.fno = FNO(
            fno_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=hidden_channels[0] + 3 + in_channels,
            out_channels=fno_out_channels,
            use_mlp=True,
            mlp={"expansion": 1.0, "dropout": 0},
            domain_padding=fno_domain_padding,
            factorization=fno_factorization,
            norm=fno_norm,
            rank=fno_rank,
        )
        self.projection = Projection(
            in_channels=hidden_channels[1],
            out_channels=out_channels,
            hidden_channels=256,
            non_linearity=paddle.nn.functional.gelu,
            n_dim=1,
        )
        self.integral_cd = Integral_Cd()
        self.print_model_size()

    def forward(self, x_in, x_out, df, x_eval=None, area_in=None, area_eval=None):
        in_to_out_nb = self.nb_search_in(x_in, x_out.reshape((-1, 3)))
        if x_eval is not None:
            out_to_in_nb = self.nb_search_out(x_out.reshape((-1, 3)), x_eval)
        else:
            out_to_in_nb = self.nb_search_out(x_out.reshape((-1, 3)), x_in)
        resolution = tuple(df.shape)[-1]
        n_in = tuple(x_in.shape)[0]
        if area_in is None or self.weighted_kernel is False:
            area_in = paddle.ones(shape=(n_in,))
        x_in = paddle.concat(x=[x_in, area_in.unsqueeze(axis=-1)], axis=-1)
        x_in_embed = self.pos_embed(x_in.reshape((-1,))).reshape((n_in, -1))
        if x_eval is not None:
            n_eval = tuple(x_eval.shape)[0]
            if area_eval is None or self.weighted_kernel is False:
                area_eval = paddle.ones(shape=(n_eval,))
            x_eval = paddle.concat(x=[x_eval, area_eval.unsqueeze(axis=-1)], axis=-1)
            x_eval_embed = self.pos_embed(x_eval.reshape((-1,))).reshape((n_eval, -1))
        x_out_embed = self.pos_embed(x_out.reshape((-1,))).reshape(
            (resolution**3, -1)
        )
        df_embed = self.df_embed(df.transpose(perm=[1, 2, 3, 0])).reshape(
            (resolution**3, -1)
        )
        grid_embed = paddle.concat(x=[x_out_embed, df_embed], axis=-1)
        u = self.gno1(x_in_embed, in_to_out_nb, grid_embed, area_in)
        u = (
            u.reshape((resolution, resolution, resolution, -1))
            .transpose(perm=[3, 0, 1, 2])
            .unsqueeze(axis=0)
        )
        u = paddle.concat(
            x=(
                x_out.transpose(perm=[3, 0, 1, 2]).unsqueeze(axis=0),
                df.unsqueeze(axis=0),
                u,
            ),
            axis=1,
        )
        u = self.fno(u)
        u = u.squeeze().transpose(perm=[1, 2, 3, 0]).reshape((resolution**3, -1))
        if self.linear_kernel == False:
            if x_eval is not None:
                u = self.gno2(u, out_to_in_nb, x_eval_embed)
            else:
                u = self.gno2(u, out_to_in_nb, x_in_embed)
        elif x_eval is not None:
            u = self.gno2(
                x_in=x_out_embed,
                neighbors=out_to_in_nb,
                in_features=u,
                x_out=x_eval_embed,
            )
        else:
            u = self.gno2(
                x_in=x_out_embed,
                neighbors=out_to_in_nb,
                in_features=u,
                x_out=x_in_embed,
            )
        u = u.unsqueeze(axis=0).transpose(perm=[0, 2, 1])
        u = self.projection(u).squeeze(axis=0).transpose(perm=[1, 0])
        return u

    def print_model_size(self):
        print("--------------------------------")
        print("The MLP_1 size is ", count_params(self.df_embed))
        print("The gno1 size is ", count_params(self.gno1))
        print("The gno2 size is ", count_params(self.gno2))
        print("The fno size is ", count_params(self.fno))
        print("The projection size is ", count_params(self.projection))
        print("The nb_search_in size is ", count_params(self.nb_search_in))
        print("The nb_search_out size is ", count_params(self.nb_search_out))
        print("The pos_embed size is ", count_params(self.pos_embed))
        return None


class GNOFNOGNO_all(GNOFNOGNO):
    def __init__(
        self,
        radius_in=0.05,
        radius_out=0.05,
        embed_dim=16,
        hidden_channels=(16, 16),
        in_channels=2,
        out_channels=[1, 3],
        fno_modes=(16, 16, 16),
        fno_hidden_channels=16,
        fno_out_channels=16,
        fno_domain_padding=0.125,
        fno_norm="ada_in",
        adain_embed_dim=64,
        fno_factorization="tucker",
        fno_rank=0.4,
        linear_kernel=True,
        weighted_kernel=True,
        max_in_points=5000,
        subsample_train=1,
        subsample_eval=1,
        out_keys=["pressure"],
    ):
        if fno_norm == "ada_in":
            init_norm = "group_norm"
        else:
            init_norm = fno_norm
        self.max_in_points = max_in_points
        self.subsample_train = subsample_train
        self.subsample_eval = subsample_eval
        self.out_keys = out_keys
        self.out_channels = out_channels
        super().__init__(
            radius_in=radius_in,
            radius_out=radius_out,
            embed_dim=embed_dim,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=sum(out_channels),
            fno_modes=fno_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_out_channels=fno_out_channels,
            fno_domain_padding=fno_domain_padding,
            fno_norm=init_norm,
            fno_factorization=fno_factorization,
            fno_rank=fno_rank,
            linear_kernel=linear_kernel,
            weighted_kernel=weighted_kernel,
        )
        if fno_norm == "ada_in":
            self.adain_pos_embed = PositionalEmbedding(adain_embed_dim)
            self.fno.fno_blocks.norm = paddle.nn.LayerList(
                sublayers=(
                    AdaIN(adain_embed_dim, fno_hidden_channels)
                    for _ in range(
                        self.fno.fno_blocks.n_norms * self.fno.fno_blocks.convs.n_layers
                    )
                )
            )
            self.use_adain = True
        else:
            self.use_adain = False
        print("The fno + adain size is ", count_params(self.fno))
        print("--------------------------------")

    def data_dict_to_input(self, data_dict, data_device=None):
        x_in = data_dict["centroids"][0]
        x_out = (
            data_dict["df_query_points"].squeeze(axis=0).transpose(perm=[1, 2, 3, 0])
        )
        df = data_dict["df"]
        area = data_dict["areas"][0]
        info_fields = data_dict["info"][0]["velocity"] * paddle.ones_like(x=df).to(
            "float32"
        )
        df = paddle.concat(x=(df, info_fields), axis=0)
        if self.use_adain:
            vel = (
                paddle.to_tensor(data=[data_dict["info"][0]["velocity"]])
                .reshape((-1,))
                .to("float32")
            )
            vel_embed = self.adain_pos_embed(vel)
            for norm in self.fno.fno_blocks.norm:
                norm.update_embeddding(vel_embed)
        return x_in, x_out, df, area

    @paddle.no_grad()
    def eval_dict(self, device, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict, device)
        x_in = x_in[:: self.subsample_eval, ...]
        area = area[:: self.subsample_eval]
        if self.max_in_points is not None:
            r = min(self.max_in_points, tuple(x_in.shape)[0])
            pred_chunks = []
            x_in_sections = [r] * (x_in.shape[0] // r)
            if x_in.shape[0] % r != 0:
                x_in_sections.append(-1)
            area_sections = [r] * (area.shape[0] // r)
            if area.shape[0] % r != 0:
                area_sections.append(-1)
            x_in_chunks = paddle.split(x=x_in, num_or_sections=x_in_sections, axis=0)
            area_chunks = paddle.split(x=area, num_or_sections=area_sections, axis=0)
            for j in range(len(x_in_chunks)):
                pred_index_j = super().forward(
                    x_in,
                    x_out,
                    df,
                    x_in_chunks[j],
                    area_in=area,
                    area_eval=area_chunks[j],
                )
                pred_chunks.append(pred_index_j)
                # paddle.device.cuda.empty_cache()  # clear GPU memory
            pred = paddle.concat(x=tuple(pred_chunks), axis=0)
        else:
            pred = self(x_in, x_out, df, area=area)
        pred = pred.transpose(perm=[1, 0])
        if loss_fn is None:
            loss_fn = self.loss
        out_dict = {
            "Cd_pred": paddle.to_tensor(data=0.0).cuda(blocking=True),
            "Cd_truth": paddle.to_tensor(data=0.0).cuda(blocking=True),
        }
        truth = []
        for i in range(len(self.out_keys)):
            key = self.out_keys[i]
            truth_key = data_dict[key][0].to(device)[:: self.subsample_eval, ...]
            # assert not paddle.any(paddle.isnan(truth_key)), "truth_key 存在无效值！"
            if len(tuple(truth_key.shape)) == 1:
                truth_key = truth_key.reshape((-1, 1))
            truth_key = truth_key[:, : self.out_channels[i]].transpose(perm=[1, 0])
            truth.append(truth_key)
            st, end = (
                sum(self.out_channels[:i]),
                sum(self.out_channels[:i]) + self.out_channels[i],
            )
            pred_key = pred[st:end, :]
            out_dict[f"L2_{key}"] = loss_fn(pred_key, truth_key)
            if decode_fn is not None:
                pred_decode = decode_fn(pred_key, i)
                truth_decode = decode_fn(truth_key, i)
                if key == "pressure":
                    drag_weight = data_dict["dragWeight"][0].cuda(blocking=True)
                    drag_weight = drag_weight[:: self.subsample_eval]
                elif key == "wallshearstress":
                    drag_weight = data_dict["dragWeightWss"][0][
                        : self.out_channels[i], :
                    ].cuda(blocking=True)
                    drag_weight = drag_weight[..., :: self.subsample_eval]
                drag_pred = paddle.sum(x=drag_weight * pred_decode)
                drag_truth = paddle.sum(x=drag_weight * truth_decode)
                out_dict.update(
                    {f"Cd_{key}_pred": drag_pred, f"Cd_{key}_truth": drag_truth}
                )
                out_dict["Cd_pred"] += drag_pred
                out_dict["Cd_truth"] += drag_truth
        truth = paddle.concat(x=truth, axis=0)

        cd_dict = {}
        cd_dict.update({"Cd_pred": out_dict["Cd_pred"]})
        cd_dict.update({"Cd_truth": out_dict["Cd_truth"]})
        cd_dict.update({"Cd_pressure_pred": out_dict["Cd_pressure_pred"]})
        cd_dict.update({"Cd_pressure_truth": out_dict["Cd_pressure_truth"]})
        cd_dict.update({"Cd_wallshearstress_pred": out_dict["Cd_wallshearstress_pred"]})
        cd_dict.update(
            {"Cd_wallshearstress_truth": out_dict["Cd_wallshearstress_truth"]}
        )
        cd_dict = self.integral_cd(cd_dict, self.out_keys)
        cd_dict.update({"L2_pressure": out_dict["L2_pressure"]})
        cd_dict.update({"L2_wallshearstress": out_dict["L2_wallshearstress"]})

        return out_dict, pred, truth, cd_dict

    @paddle.no_grad()
    def inference_dict(self, device, data_dict, loss_fn=None, decode_fn=None, **kwargs):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict, device)
        x_in = x_in[:: self.subsample_eval, ...]
        area = area[:: self.subsample_eval]
        if self.max_in_points is not None:
            r = min(self.max_in_points, tuple(x_in.shape)[0])
            pred_chunks = []
            x_in_sections = [r] * (x_in.shape[0] // r)
            if x_in.shape[0] % r != 0:
                x_in_sections.append(-1)
            area_sections = [r] * (area.shape[0] // r)
            if area.shape[0] % r != 0:
                area_sections.append(-1)
            x_in_chunks = paddle.split(x=x_in, num_or_sections=x_in_sections, axis=0)
            area_chunks = paddle.split(x=area, num_or_sections=area_sections, axis=0)
            for j in range(len(x_in_chunks)):
                # t = time.perf_counter()
                pred_index_j = super().forward(
                    x_in,
                    x_out,
                    df,
                    x_in_chunks[j],
                    area_in=area,
                    area_eval=area_chunks[j],
                )
                pred_chunks.append(pred_index_j)
                # paddle.device.cuda.empty_cache()  # clear GPU memory
                # t = time.perf_counter() - t
                # print(f"chunk {j}: {t:.3f} s")
            pred = paddle.concat(x=tuple(pred_chunks), axis=0)
        else:
            pred = self(x_in, x_out, df, area=area)
        pred = pred.transpose(perm=[1, 0])
        if loss_fn is None:
            loss_fn = self.loss
        out_dict = {"Cd_pred": paddle.to_tensor(data=0.0).cuda(blocking=True)}

        for i in range(len(self.out_keys)):
            key = self.out_keys[i]
            # assert not paddle.any(paddle.isnan(truth_key)), "truth_key 存在无效值！"

            st, end = (
                sum(self.out_channels[:i]),
                sum(self.out_channels[:i]) + self.out_channels[i],
            )
            pred_key = pred[st:end, :]
            if decode_fn is not None:
                pred_decode = decode_fn(pred_key, i)
                if key == "pressure":
                    drag_weight = data_dict["dragWeight"][0].cuda(blocking=True)
                    drag_weight = drag_weight[:: self.subsample_eval]
                elif key == "wallshearstress":
                    drag_weight = data_dict["dragWeightWss"][0][
                        : self.out_channels[i], :
                    ].cuda(blocking=True)
                    drag_weight = drag_weight[..., :: self.subsample_eval]
                drag_pred = paddle.sum(x=drag_weight * pred_decode)
                out_dict.update({f"Cd_{key}_pred": drag_pred})
                out_dict["Cd_pred"] += drag_pred

        cd_dict = {}
        cd_dict.update({"Cd_pred": out_dict["Cd_pred"].item()})
        cd_dict.update({"Cd_pressure_pred": out_dict["Cd_pressure_pred"].item()})
        cd_dict.update(
            {"Cd_wallshearstress_pred": out_dict["Cd_wallshearstress_pred"].item()}
        )
        cd_dict = self.integral_cd(cd_dict, self.out_keys)

        velocity = data_dict["info"][0]["velocity"]
        reference_area = data_dict["info"][0]["reference_area"]
        density = data_dict["info"][0]["density"]
        const = 0.5 * density * velocity**2 * reference_area

        cd_dict.update({"total_drag_pred": const * cd_dict["Cd_pred_modify"].item()})

        cd_dict.update(
            {
                "pressure_drag_pred": const
                * (
                    cd_dict["Cd_pressure_pred"]
                    + cd_dict["Cd_pred_modify"].item()
                    - out_dict["Cd_pred"].item()
                )
            }
        )

        cd_dict.update(
            {
                "wallshearstress_drag_pred": const
                * out_dict["Cd_wallshearstress_pred"].item()
            }
        )

        return out_dict, pred, cd_dict

    def forward(
        self,
        data_dict,
        idx_batch,
        device=None,
        randperm=True,
        loss_fn=None,
        decode_fn=None,
    ):
        x_in, x_out, df, area = self.data_dict_to_input(data_dict, device)
        x_in = x_in[:: self.subsample_train, ...]
        area = area[:: self.subsample_train]
        r = min(self.max_in_points, tuple(x_in.shape)[0])
        if randperm:
            indices = paddle.randperm(n=tuple(x_in.shape)[0])[:r]
        else:
            indices = paddle.linspace(
                start=0, stop=tuple(x_in.shape)[0] - 1, num=r, dtype="int64"
            ).astype("int64")

        truth = []
        for i in range(len(self.out_keys)):
            truth_key = data_dict[self.out_keys[i]][0][:: self.subsample_train]
            if len(tuple(truth_key.shape)) == 1:
                truth_key = truth_key.reshape([-1, 1])
            truth_key = truth_key[indices][:, : self.out_channels[i]].to(x_in.place)
            truth.append(truth_key)
        truth = paddle.concat(x=truth, axis=-1)

        if self.integral_cd.parameters()[0].stop_gradient == True:
            pred = super().forward(
                x_in, x_out, df, x_in[indices, ...], area, area[indices]
            )
        else:
            pred = truth
            # paddle.device.cuda.empty_cache()  # clear GPU memory

        cd_dict = {}
        if self.integral_cd.parameters()[0].stop_gradient == False:
            # cd_dict = self.integral_cd(pred, truth, self.out_channels,
            #    data_dict, decode_fn=decode_fn,
            #    out_keys=self.out_keys,
            #    subsample_train=self.subsample_train)

            cd_dict.update({"OOM": False})
            try:
                out_dict, _, _, _ = self.eval_dict(
                    device, data_dict, loss_fn=loss_fn, decode_fn=decode_fn
                )

                cd_dict.update({"Cd_pred": out_dict["Cd_pred"]})
                cd_dict.update({"Cd_truth": out_dict["Cd_truth"]})
                cd_dict.update({"Cd_pressure_pred": out_dict["Cd_pressure_pred"]})
                cd_dict.update({"Cd_pressure_truth": out_dict["Cd_pressure_truth"]})
                cd_dict.update(
                    {"Cd_wallshearstress_pred": out_dict["Cd_wallshearstress_pred"]}
                )
                cd_dict.update(
                    {"Cd_wallshearstress_truth": out_dict["Cd_wallshearstress_truth"]}
                )
                cd_dict = self.integral_cd(cd_dict, self.out_keys)
                cd_dict.update({"L2_pressure": out_dict["L2_pressure"]})
                cd_dict.update({"L2_wallshearstress": out_dict["L2_wallshearstress"]})

                def cal_mre(pred, label):
                    return paddle.abs(x=pred - label) / paddle.abs(x=label)

                # print(f'sample {idx_batch} cd_dict:', cd_dict)

            except MemoryError as e:
                if "Out of memory" in str(e):
                    print(f"WARNING: OOM on sample {idx_batch}, skipping this sample.")
                    if hasattr(paddle.device.cuda, "empty_cache"):
                        paddle.device.cuda.empty_cache()
                    cd_dict.update({"OOM": True})
                else:
                    raise

        return pred.transpose(perm=[1, 0]), truth.transpose(perm=[1, 0]), cd_dict
