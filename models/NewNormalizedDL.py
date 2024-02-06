#@title **Clean Mod Version something to push ?**
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride,avgType=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        if(avgType == 1):
          self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        else:
          self.avg = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        # self.avg = nn.Conv1d(in_channels = 1,out_channels=1,kernel_size=kernel_size, stride=stride, padding=0)
        # self.avg = nn.AdaptiveMaxPool1d(output_size=336)
        # self.avg = nn.LPPool1d(2,kernel_size=kernel_size, stride=stride)

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
    def __init__(self, kernel_size,avgType=2):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1,avgType=avgType)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        # moving_mean = -self.moving_avg(-x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    New model-
    """
    def __init__(self, configs,kernel_size=[25]):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch_size = configs.batch_size

        self.individual = configs.individual
        self.channels = configs.enc_in

        self.decompsition = series_decomp(25,2)
        self.decompsition1 = series_decomp(25,1)


        self.SelectorDistance = torch.nn.Parameter(torch.tensor([0.1]))
        self.SelectorScaler = torch.nn.Parameter(torch.tensor([0.25]))

        self.ScalerParam1 = torch.nn.Parameter(torch.tensor([0.25]))
        self.ScalerParam2 = torch.nn.Parameter(torch.randn((1,)))
        self.ScalerParam3 = torch.nn.Parameter(torch.randn((1,)))
        self.ScalerParam4 = torch.nn.Parameter(torch.randn((1,)))
        self.ScalerParams = torch.nn.Parameter(torch.randn((4,)))


        self.LN = torch.nn.Parameter(torch.tensor([0.25]))
        self.LN1 = torch.nn.Parameter(torch.tensor([1.]))
        self.LN2 = torch.nn.Parameter(torch.tensor([0.25]))
        # self.LN2 = torch.nn.Parameter(torch.tensor([2.94]))
        self.relu = nn.ReLU()

        self.LS = torch.nn.Parameter(torch.tensor([0.25]))

        self.TimeStampsSeq = nn.Sequential(
                  nn.Linear(self.seq_len,self.seq_len),
                  # nn.ReLU(),
                  # nn.Linear(self.seq_len,self.seq_len),
                  # nn.GELU(),
                  # nn.Linear(self.seq_len,self.seq_len),
              )

        self.SeasonalSeq =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
              )

        self.TrendSeq =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
              )

        self.SeasonalSeq1 =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
              )

        self.TrendSeq1 =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
              )


        self.ResultSeq =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
              )
        self.LG =nn.Sequential(
                  nn.Linear(self.seq_len,self.seq_len),
                  nn.GELU(),
                  nn.Linear(self.seq_len,1),
              )
        self.LG1 =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
                  # nn.GELU(),
                  # nn.Linear(self.pred_len,self.pred_len),
              )
        self.LG2 =nn.Sequential(
                  nn.Linear(self.pred_len,self.pred_len),
                  # nn.GELU(),
                  # nn.Linear(self.pred_len,self.pred_len),
              )

    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark,y=None):
        # x: [Batch, Input length, Channel]
        s, t = self.decompsition1(x)

        # Normalisation
        seq_last = x[:,-1:,:].detach()
        # seq_last = t[:,-1:,:].detach()
        x = x - seq_last
        z = x.detach()
        # Selector
        selector = torch.zeros([x.shape[0],self.seq_len,x.shape[2]],dtype=x.dtype).to(x.device)
        # population = x.mean() * (8126 / (8126 - 4063))
        # population = x.mean() * ((self.batch_size*self.seq_len)/((self.batch_size*self.seq_len)-(self.batch_size*self.pred_len)))
        # population = x.mean() * ((22*self.seq_len*self.batch_size) / ((22*self.seq_len*self.batch_size)*x.std()))
        p = self.LG(x.permute(0,2,1)).sum(0).squeeze()
        equalizer = np.sqrt(((self.seq_len+self.pred_len)*self.batch_size)/((self.batch_size*(self.pred_len+self.seq_len))-(self.batch_size*(self.seq_len))))
        population = x.mean() * equalizer
        condition = population + self.SelectorDistance

        selector[abs(x)>condition] = self.SelectorScaler

        x = selector * x


        # timestamps embedding
        timestamps = self.TimeStampsSeq(batch_x_mark.permute(0,2,1)).permute(0,2,1)

        timestamps = (timestamps[:,:,0] * self.ScalerParam1 +
                      timestamps[:,:,1] * self.ScalerParam2 +
                      timestamps[:,:,2] * self.ScalerParam3 +
                      timestamps[:,:,3] * self.ScalerParam4 ).unsqueeze(2)


        # timestamps = torch.sum(timestamps * self.ScalerParams.view(1, 1, 4), dim=2, keepdim=True)


        # DLinear
        seasonal_init, trend_init = self.decompsition(x)

        seasonal_out = self.SeasonalSeq(seasonal_init.permute(0,2,1)).permute(0,2,1)
        trend_out = self.TrendSeq(trend_init.permute(0,2,1)).permute(0,2,1)

        seasonal_init1, trend_init1 = self.decompsition(z)

        seasonal_out1 = self.SeasonalSeq1(seasonal_init1.permute(0,2,1)).permute(0,2,1)
        trend_out1 = self.TrendSeq1(trend_init1.permute(0,2,1)).permute(0,2,1)


        decide = self.relu(torch.sin(self.LN1))

        # decide1 = self.relu(torch.sin(self.LN1))

        seasonal_out = seasonal_out*self.LN + seasonal_out1 * (1-self.LN)

        # trend_out = trend_out*self.LN1 + trend_out1 * (1-self.LN1)

        # forecast = seasonal_out + trend_out

        forecast = seasonal_out1 + trend_out #* self.LN1

        # Results
        y = self.ResultSeq(x.permute(0,2,1) * timestamps ).permute(0,2,1).mean(axis=2).unsqueeze(2)

        timestamps = self.LG1(timestamps.permute(0,2,1) ).permute(0,2,1)

        y = y * forecast   + seq_last

        # y = y.mean(axis=2).unsqueeze(2)

        return y # to [Batch, Output length, Channel]




# DLinear = {
#     "Model":Model
# }
# DLinear = AttributeDict(DLinear)
