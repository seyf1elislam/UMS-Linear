import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
# 

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        # self.conv = conv(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    this model mix of Nlienar and DLinear with Timestamp embeding and
    """
    def __init__(self, configs,kernel_size=[25]):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.individual = configs.individual
        self.channels = configs.enc_in

        self.decompsition = series_decomp(25)

        self.SelectorDistance = torch.nn.Parameter(torch.tensor([0.1]))
        self.SelectorScaler = torch.nn.Parameter(torch.tensor([0.25]))

        # self.ScalerParam1 = torch.nn.Parameter(torch.tensor([0.25]))
        # self.ScalerParam2 = torch.nn.Parameter(torch.randn((1,)))
        # self.ScalerParam3 = torch.nn.Parameter(torch.randn((1,)))
        # self.ScalerParam4 = torch.nn.Parameter(torch.randn((1,)))
        # self.ScalerParams = torch.nn.Parameter(torch.tensor([0.25, 1., 1., 1.]))
        self.ScalerParams = torch.nn.Parameter(torch.randn(4))

        self.TimeStampsSeq = nn.Sequential(
                  nn.Linear(self.seq_len,self.seq_len),
                  # nn.GELU(),
                  # nn.Linear(self.seq_len,self.seq_len),
              )

        self.SeasonalSeq =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
              )

        self.TrendSeq =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
              )

        self.ResultSeq =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
              )


    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark,y=None):
        # x: [Batch, Input length, Channel] (input length is  seq_len)

        # Normalisation =========================================================================
        seq_last = x[:,-1:,:].detach() #seq_last: [Batch, 1, Channel]
        
        x_normalized = x - seq_last #
        

        # Selector =========================================================================
        # selector: [Batch, Input length, Channel]
        selector = torch.zeros([x.shape[0],self.seq_len,x.shape[2]],dtype=x.dtype).to(x.device)
        #condition :size[1] , selector distance [1] ,selector scaler [1]
        condition = x_normalized.mean() + self.SelectorDistance
        # condition =self.SelectorDistance

        selector[abs(x_normalized)>condition] = self.SelectorScaler
        # x_normalized_selected: [Batch, Input length, Channel]
        x_normalized_selected = selector * x_normalized
        

        # timestamps embedding =========================================================================
        # batch_x_mark: [Batch, Input length, 4] 4 YY/MM/YY/HH
        # timestamps| embedded timestamp: [Batch, pred_length, 4] 4 YY/MM/YY/HH

        timestamps = self.TimeStampsSeq(batch_x_mark.permute(0,2,1)).permute(0,2,1)

        # timestamps = (timestamps[:,:,0] * self.ScalerParam1 +
        #               timestamps[:,:,1] * self.ScalerParam2 +
        #               timestamps[:,:,2] * self.ScalerParam3 +
        #               timestamps[:,:,3] * self.ScalerParam4 ).unsqueeze(2)
        # print(f"timestamps shape :{timestamps.shape}")
        # print(f"ScalerParams shape :{self.ScalerParams.shape}")
        #scalerParams :size[4]
        # timestamps = torch.sum(timestamps * self.ScalerParams.view(1, 1, 4), dim=2, keepdim=True)#same with the two sqeez in chain
        timestamps = torch.sum(timestamps * self.ScalerParams.unsqueeze(0).unsqueeze(0), dim=2, keepdim=True)
        


        # DLinear =========================================================================
        # seasonal_init/trend init size : [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x_normalized_selected)
        #season_out/trend out : : [Batch, prediction length, Channel]
        seasonal_out = self.SeasonalSeq(seasonal_init.permute(0,2,1)).permute(0,2,1)
        trend_out = self.TrendSeq(trend_init.permute(0,2,1)).permute(0,2,1)

        forecast = seasonal_out+trend_out

        # Results
        y = self.ResultSeq(x_normalized_selected.permute(0,2,1) * timestamps).permute(0,2,1).mean(axis=1).unsqueeze(1)

        y = y * forecast  + seq_last
        return y # to [Batch, Output length, Channel]
