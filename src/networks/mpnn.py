import torch
import torch_geometric

import numpy

from src.config.network import GrowthRate, DownSampling
from src.config.framework import DataMode


class MLP(torch.nn.Module):

    def __init__(self, n_in, params):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        for n_out in params.layers:
            self.layers.append(torch.nn.Linear(n_in, n_out, bias=params.bias))
            self.layers.append(torch.nn.Tanh())
            n_in = n_out

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(torch.nn.Module):

    def __init__(self, params, image_size):
        super().__init__()

        print(params.encoder)

        n_init = params.encoder.n_initial_filters
        self.feature_encoder = torch.nn.Linear(
            in_features=4,
            out_features=(n_init - 4),
            bias=True
        )

        self.layers = torch.nn.ModuleList()

        n_out_total = 0


        n_in = n_init
        for i_layer in range(params.encoder.depth):
            mlp = MLP(n_in, params.encoder.mlp_config)
            # The total number of outputs will increase with this layer:
            n_out_total += params.encoder.mlp_config.layers[-1]
            self.layers.append( torch_geometric.nn.conv.GINConv(mlp) )

        # self.model = torch_geometric.nn.models.GAT(in_channels=-1,
        #                                            hidden_channels=8,
        #                                            num_layers=2)


        self.output_shape = (n_out_total,)
        # self.initial_layer = torch_geometric.nn.MessagePassing()

    def forward(self, batch):

        # First, update the initial features:
        x = batch.x
        x_aug = self.feature_encoder(x)
        x = torch.concat([x, x_aug], axis=-1)

        edges = batch.edge_index


        encoding_list = []

        for layer in self.layers:
            new_nodes = layer(x, edges)
            encoding_list.append(torch_geometric.nn.pool.global_mean_pool(new_nodes, batch=batch.batch))
        
        encoding = torch.concatenate(encoding_list, axis=-1)

        return encoding
        # return encoding.reshape((-1,) + self.output_shape)