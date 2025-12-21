# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Graph Network models for physics discovery.

This module implements three types of graph networks:
- OGN: Object-based Graph Network (Newtonian mechanics)
- HGN: Hamiltonian Graph Network (Energy-based formulation)
- VarOGN: Variational OGN (Uncertainty quantification)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import paddle
import paddle.nn as nn


def get_edge_index(n: int, sim: str) -> np.ndarray:
    """
    Generate edge indices for graph connectivity.

    Args:
        n (int): Number of nodes.
        sim (str): Simulation type.

    Returns:
        numpy.ndarray: Edge indices with shape [2, num_edges].
    """
    if sim in ["string", "string_ball"]:
        # Chain topology for string simulations
        top = np.arange(0, n - 1)
        bottom = np.arange(1, n)
        edge_index = np.concatenate(
            [
                np.concatenate([top, bottom])[None, :],
                np.concatenate([bottom, top])[None, :],
            ],
            axis=0,
        )
    else:
        # Fully connected graph (all-to-all)
        adj = (np.ones((n, n)) - np.eye(n)).astype(int)
        edge_index = np.array(np.where(adj))

    return edge_index


class OGN(nn.Layer):
    """
    Object-based Graph Network (OGN).

    Models particle dynamics using message passing neural networks.
    Predicts acceleration based on Newtonian mechanics (F=ma).
    """

    def __init__(
        self,
        n_f: int = 6,
        msg_dim: int = 100,
        ndim: int = 2,
        hidden: int = 300,
        edge_index: Optional[np.ndarray] = None,
        l1_strength: float = 0.0,
    ):
        """
        Initialize OGN model.

        Args:
            n_f: Node feature dimension (default: 6 for 2D: x,y,vx,vy,charge,mass).
            msg_dim: Message dimension.
            ndim: Spatial dimension (2 or 3).
            hidden: Hidden layer size.
            edge_index: Fixed edge indices (optional).
            l1_strength: L1 regularization strength for sparsity.
        """
        super().__init__()

        self.n_f = n_f
        self.msg_dim = msg_dim
        self.ndim = ndim
        self.hidden = hidden
        self.l1_strength = l1_strength

        if edge_index is not None:
            self.register_buffer(
                "edge_index_buffer", paddle.to_tensor(edge_index, dtype="int64")
            )
        else:
            self.edge_index_buffer = None

        # Message function: (x_i, x_j) -> message
        self.msg_fnc = nn.Sequential(
            nn.Linear(2 * n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, msg_dim),
        )

        # Node update function: (x_i, aggregated_messages) -> acceleration
        self.node_fnc = nn.Sequential(
            nn.Linear(msg_dim + n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ndim),
        )

    def message_passing(
        self, x: paddle.Tensor, edge_index: np.ndarray
    ) -> paddle.Tensor:
        """
        Execute message passing to predict acceleration.

        Args:
            x: Node features [batch, n_nodes, n_f] or [n_nodes, n_f].
            edge_index: Edge indices [2, num_edges].

        Returns:
            paddle.Tensor: Predicted acceleration [batch, n_nodes, ndim] or [n_nodes, ndim].
        """
        if len(x.shape) == 3:
            # Batch mode
            batch_size, n, n_f = x.shape
            x_reshaped = x.reshape([-1, n_f])

            results = []
            all_messages = []
            for b in range(batch_size):
                start_idx = b * n
                end_idx = (b + 1) * n
                x_batch = x_reshaped[start_idx:end_idx]

                row, col = edge_index[0], edge_index[1]

                # Get source and target node features
                x_i = x_batch[col]  # Receiver
                x_j = x_batch[row]  # Sender

                # Compute messages
                msg_input = paddle.concat([x_i, x_j], axis=1)
                msg = self.msg_fnc(msg_input)
                all_messages.append(msg)

                # Aggregate messages (sum)
                aggr_out = paddle.zeros([n, self.msg_dim], dtype=msg.dtype)
                for i in range(len(col)):
                    aggr_out[col[i]] += msg[i]

                # Update nodes
                node_input = paddle.concat([x_batch, aggr_out], axis=1)
                out = self.node_fnc(node_input)
                results.append(out)

            return paddle.stack(results, axis=0), paddle.concat(all_messages, axis=0)

        else:
            # Single sample mode
            row, col = edge_index[0], edge_index[1]

            x_i = x[col]
            x_j = x[row]

            msg_input = paddle.concat([x_i, x_j], axis=1)
            msg = self.msg_fnc(msg_input)

            num_nodes = x.shape[0]
            aggr_out = paddle.zeros([num_nodes, self.msg_dim], dtype=msg.dtype)

            for i in range(len(col)):
                aggr_out[col[i]] += msg[i]

            node_input = paddle.concat([x, aggr_out], axis=1)
            out = self.node_fnc(node_input)

            return out, msg

    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        """
        Forward propagation.

        Args:
            inputs: Input dictionary containing:
                - 'x': Node features [batch, n_nodes, n_f] or [n_nodes, n_f]
                - 'edge_index': Edge indices (optional if provided at initialization)

        Returns:
            Dict containing 'acceleration' and optionally 'l1_regularization'.
        """
        x = inputs["x"]

        if "edge_index" in inputs:
            edge_index = inputs["edge_index"]
            if isinstance(edge_index, paddle.Tensor):
                edge_index = edge_index.numpy()
            if len(edge_index.shape) == 3:
                edge_index = edge_index[0]
        elif self.edge_index_buffer is not None:
            edge_index = self.edge_index_buffer.numpy()
        else:
            raise ValueError("Must provide edge_index")

        acceleration, messages = self.message_passing(x, edge_index)

        outputs = {"acceleration": acceleration}

        # L1 regularization for sparsity on message vectors
        if self.l1_strength > 0:
            l1_reg = self.l1_strength * paddle.mean(paddle.abs(messages))
            outputs["l1_regularization"] = l1_reg

        return outputs


class HGN(nn.Layer):
    """
    Hamiltonian Graph Network (HGN).

    Models conservative systems using Hamiltonian mechanics.
    Predicts dynamics via Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q.
    """

    def __init__(
        self,
        n_f: int = 6,
        ndim: int = 2,
        hidden: int = 300,
        edge_index: Optional[np.ndarray] = None,
    ):
        """
        Initialize HGN model.

        Args:
            n_f: Node feature dimension.
            ndim: Spatial dimension.
            hidden: Hidden layer size.
            edge_index: Fixed edge indices (optional).
        """
        super().__init__()

        self.n_f = n_f
        self.ndim = ndim
        self.hidden = hidden

        if edge_index is not None:
            self.register_buffer(
                "edge_index_buffer", paddle.to_tensor(edge_index, dtype="int64")
            )
        else:
            self.edge_index_buffer = None

        # Pairwise energy function
        self.pair_energy = nn.Sequential(
            nn.Linear(2 * n_f, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

        # Self energy function
        self.self_energy = nn.Sequential(
            nn.Linear(n_f, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def compute_energy(self, x: paddle.Tensor, edge_index: np.ndarray) -> paddle.Tensor:
        """
        Compute total Hamiltonian energy.

        Args:
            x: Node features in Hamiltonian form [q, p, other].
            edge_index: Edge indices.

        Returns:
            paddle.Tensor: Energy per node.
        """
        row, col = edge_index[0], edge_index[1]

        x_i = x[col]
        x_j = x[row]
        edge_input = paddle.concat([x_i, x_j], axis=1)
        pair_energies = self.pair_energy(edge_input)

        num_nodes = x.shape[0]
        aggr_pair = paddle.zeros([num_nodes, 1], dtype=pair_energies.dtype)
        for i in range(len(col)):
            aggr_pair[col[i]] += pair_energies[i]

        self_energies = self.self_energy(x)

        total_energy = aggr_pair + self_energies

        return total_energy

    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        """
        Forward propagation using Hamilton's equations.

        Args:
            inputs: Input dictionary containing:
                - 'x': Node features [q, v, other] where q=position, v=velocity
                - 'edge_index': Edge indices (optional)

        Returns:
            Dict containing 'acceleration'.
        """
        x_input = inputs["x"].clone()

        if "edge_index" in inputs:
            edge_index = inputs["edge_index"]
            if isinstance(edge_index, paddle.Tensor):
                edge_index = edge_index.numpy()
            if len(edge_index.shape) == 3:
                edge_index = edge_index[0]
        elif self.edge_index_buffer is not None:
            edge_index = self.edge_index_buffer.numpy()
        else:
            raise ValueError("Must provide edge_index")

        if len(x_input.shape) == 3:
            # Batch mode
            batch_size, n, n_f = x_input.shape
            x_reshaped = x_input.reshape([-1, n_f])

            results = []
            for b in range(batch_size):
                start_idx = b * n
                end_idx = (b + 1) * n
                x_batch = x_reshaped[start_idx:end_idx]

                # Convert (q, v, other) to (q, p, other)
                q = x_batch[:, : self.ndim]
                v = x_batch[:, self.ndim : 2 * self.ndim]
                other = x_batch[:, 2 * self.ndim :]

                m_scalar = other[:, -1:]  # Mass
                m_vec = paddle.tile(m_scalar, [1, self.ndim])

                p = v * m_vec  # Momentum

                x_hamilton = paddle.concat([q, p, other], axis=1)
                x_hamilton.stop_gradient = False

                # Compute Hamiltonian
                total_energy = self.compute_energy(x_hamilton, edge_index)
                total_energy_scalar = paddle.sum(total_energy)

                # Hamilton's equations via autodiff
                dH = paddle.grad(
                    outputs=total_energy_scalar,
                    inputs=x_hamilton,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                dH_dq = dH[:, : self.ndim]
                dH_dp = dH[:, self.ndim : 2 * self.ndim]

                dq_dt = dH_dp  # Velocity
                dp_dt = -dH_dq  # Force
                dv_dt = dp_dt / m_vec  # Acceleration

                derivative = paddle.concat([dq_dt, dv_dt], axis=1)
                results.append(derivative)

            derivative = paddle.stack(results, axis=0)
            acceleration = derivative[:, :, self.ndim :]  # Extract acceleration part

        else:
            # Single sample mode
            q = x_input[:, : self.ndim]
            v = x_input[:, self.ndim : 2 * self.ndim]
            other = x_input[:, 2 * self.ndim :]

            m_scalar = other[:, -1:]
            m_vec = paddle.tile(m_scalar, [1, self.ndim])

            p = v * m_vec

            x_hamilton = paddle.concat([q, p, other], axis=1)
            x_hamilton.stop_gradient = False

            total_energy = self.compute_energy(x_hamilton, edge_index)
            total_energy_scalar = paddle.sum(total_energy)

            dH = paddle.grad(
                outputs=total_energy_scalar,
                inputs=x_hamilton,
                create_graph=False,
                retain_graph=False,
            )[0]

            dH_dq = dH[:, : self.ndim]
            dH_dp = dH[:, self.ndim : 2 * self.ndim]

            dq_dt = dH_dp
            dp_dt = -dH_dq
            dv_dt = dp_dt / m_vec

            acceleration = dv_dt

        return {"acceleration": acceleration}


class VarOGN(nn.Layer):
    """
    Variational Object-based Graph Network (VarOGN).

    Extension of OGN with uncertainty quantification.
    Uses variational inference to estimate prediction uncertainty.
    """

    def __init__(
        self,
        n_f: int = 6,
        msg_dim: int = 100,
        ndim: int = 2,
        hidden: int = 300,
        edge_index: Optional[np.ndarray] = None,
        enable_sampling: bool = True,
        l1_strength: float = 0.0,
    ):
        """
        Initialize VarOGN model.

        Args:
            n_f: Node feature dimension.
            msg_dim: Message dimension.
            ndim: Spatial dimension.
            hidden: Hidden layer size.
            edge_index: Fixed edge indices (optional).
            enable_sampling: Enable stochastic sampling during training.
            l1_strength: L1 regularization strength.
        """
        super().__init__()

        self.n_f = n_f
        self.msg_dim = msg_dim
        self.ndim = ndim
        self.hidden = hidden
        self.enable_sampling = enable_sampling
        self.l1_strength = l1_strength

        if edge_index is not None:
            self.register_buffer(
                "edge_index_buffer", paddle.to_tensor(edge_index, dtype="int64")
            )
        else:
            self.edge_index_buffer = None

        # Message function outputs both mean and log-variance
        self.msg_fnc = nn.Sequential(
            nn.Linear(2 * n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, msg_dim * 2),  # *2 for mu and logvar
        )

        # Node update function
        self.node_fnc = nn.Sequential(
            nn.Linear(msg_dim + n_f, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ndim),
        )

    def message_passing(
        self, x: paddle.Tensor, edge_index: np.ndarray
    ) -> paddle.Tensor:
        """
        Variational message passing with stochastic sampling.

        Args:
            x: Node features.
            edge_index: Edge indices.

        Returns:
            paddle.Tensor: Predicted acceleration (mean).
        """
        if len(x.shape) == 3:
            # Batch mode
            batch_size, n, n_f = x.shape
            x_reshaped = x.reshape([-1, n_f])

            results = []
            all_messages = []
            for b in range(batch_size):
                start_idx = b * n
                end_idx = (b + 1) * n
                x_batch = x_reshaped[start_idx:end_idx]

                row, col = edge_index[0], edge_index[1]

                x_i = x_batch[col]
                x_j = x_batch[row]

                msg_input = paddle.concat([x_i, x_j], axis=1)
                raw_msg = self.msg_fnc(msg_input)

                # Split into mean and log-variance
                mu = raw_msg[:, 0::2]
                logvar = raw_msg[:, 1::2]

                # Reparameterization trick
                if self.enable_sampling and self.training:
                    epsilon = paddle.randn(mu.shape)
                    msg = mu + epsilon * paddle.exp(logvar / 2.0)
                else:
                    msg = mu

                all_messages.append(msg)

                # Aggregate messages
                aggr_out = paddle.zeros([n, self.msg_dim], dtype=msg.dtype)
                for i in range(len(col)):
                    aggr_out[col[i]] += msg[i]

                # Update nodes
                node_input = paddle.concat([x_batch, aggr_out], axis=1)
                out = self.node_fnc(node_input)
                results.append(out)

            return paddle.stack(results, axis=0), paddle.concat(all_messages, axis=0)

        else:
            # Single sample mode
            row, col = edge_index[0], edge_index[1]

            x_i = x[col]
            x_j = x[row]

            msg_input = paddle.concat([x_i, x_j], axis=1)
            raw_msg = self.msg_fnc(msg_input)

            mu = raw_msg[:, 0::2]
            logvar = raw_msg[:, 1::2]

            if self.enable_sampling and self.training:
                epsilon = paddle.randn(mu.shape)
                msg = mu + epsilon * paddle.exp(logvar / 2.0)
            else:
                msg = mu

            num_nodes = x.shape[0]
            aggr_out = paddle.zeros([num_nodes, self.msg_dim], dtype=msg.dtype)

            for i in range(len(col)):
                aggr_out[col[i]] += msg[i]

            node_input = paddle.concat([x, aggr_out], axis=1)
            out = self.node_fnc(node_input)

            return out, msg

    def forward(self, inputs: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        """
        Forward propagation.

        Args:
            inputs: Input dictionary containing:
                - 'x': Node features
                - 'edge_index': Edge indices (optional)

        Returns:
            Dict containing 'acceleration' and optionally 'l1_regularization'.
        """
        x = inputs["x"]

        if "edge_index" in inputs:
            edge_index = inputs["edge_index"]
            if isinstance(edge_index, paddle.Tensor):
                edge_index = edge_index.numpy()
            if len(edge_index.shape) == 3:
                edge_index = edge_index[0]
        elif self.edge_index_buffer is not None:
            edge_index = self.edge_index_buffer.numpy()
        else:
            raise ValueError("Must provide edge_index")

        acceleration, messages = self.message_passing(x, edge_index)

        outputs = {"acceleration": acceleration}

        # L1 regularization on message vectors for sparsity
        if self.l1_strength > 0:
            l1_reg = self.l1_strength * paddle.mean(paddle.abs(messages))
            outputs["l1_regularization"] = l1_reg

        return outputs
