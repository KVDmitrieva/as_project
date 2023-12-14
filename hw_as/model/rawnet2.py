import torch
import torch.nn as nn

from hw_as.model.base_model import BaseModel
from hw_as.model.utils import *


class RawNet2(BaseModel):
    def __init__(self, sinc_params, resblocks_params, gru_params, linear_params, negative_slope=0.3, use_norm=True):
        super().__init__()

        self.sinc_layer = nn.Sequential(
            SincConv_fast(**sinc_params),
            nn.MaxPool1d(3),
            nn.BatchNorm1d(num_features=sinc_params["out_channels"]),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

        self.resblocks = nn.Sequential(*[ResBlock(**params) for params in resblocks_params])
        self.pregru = nn.Sequential(nn.BatchNorm1d(gru_params["input_size"]),
                                    nn.LeakyReLU(negative_slope)) if use_norm else nn.Identity()
        self.gru = nn.GRU(**gru_params)
        self.epilog = nn.Sequential(
            nn.Linear(**linear_params),
            nn.Softmax(dim=-1)
            nn.Linear(linear_params["out_features"], 2)
        )

    def forward(self, x, **batch):
        x = self.sinc_layer(x)
        x = self.resblocks(x)
        x = self.pregru(x)
        x, _ = self.gru(x.transpose(1, 2))
        x = self.epilog(x[:, -1])
        return x
