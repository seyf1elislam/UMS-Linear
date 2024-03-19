import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MinPool1d(nn.Module):
  def __init__(self, kernel_size, stride,padding):
    super(MinPool1d, self).__init__()
    self.pool=nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
  def forward(self,x):
    return -self.pool(-x)

class MaxBeforePool1d(nn.Module):
  def __init__(self, kernel_size, stride,padding,seq_len=336):
    super(MaxBeforePool1d, self).__init__()
    self.kernel_size = kernel_size
    self.seq_len = seq_len
    self.pool=nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
  def forward(self,x):
    x = x.permute(0, 2, 1)
    result = torch.zeros([x.size(0),self.seq_len,x.size(2)],dtype=x.dtype).to(x.device)
    for i in range(self.seq_len):
      e = x[:,max(0,i-self.kernel_size):i+1,:]
      result[:,i,:],_ = e.max(dim=1)
    return result.permute(0, 2, 1)

class MinBeforePool1d(nn.Module):
  def __init__(self, kernel_size, stride,padding,seq_len=336):
    super(MinBeforePool1d, self).__init__()
    self.kernel_size = kernel_size
    self.seq_len = seq_len
  def forward(self,x):
    x = x.permute(0, 2, 1)
    result = torch.zeros([x.size(0),self.seq_len,x.size(2)],dtype=x.dtype).to(x.device)
    for i in range(self.seq_len):
      e = x[:,max(0,i-self.kernel_size):i+1,:]
      result[:,i,:],_ = e.min(dim=1)
    return result.permute(0, 2, 1)

class AvgBeforePool1d(nn.Module):
  def __init__(self, kernel_size, stride,padding,seq_len=336):
    super(AvgBeforePool1d, self).__init__()
    self.kernel_size = kernel_size
    self.seq_len = seq_len
  def forward(self,x):
    x = x.permute(0, 2, 1)
    result = torch.zeros([x.size(0),self.seq_len,x.size(2)],dtype=x.dtype).to(x.device)
    for i in range(self.seq_len):
      e = x[:,max(0,i-self.kernel_size):i+1,:]
      result[:,i,:] = e.mean(dim=1)
    return result.permute(0, 2, 1)

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride,avgType=1,seq_len=336):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        if(avgType == 1):#avg
          self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        elif(avgType == 2): #max
          self.avg = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        elif(avgType == 3): #min
          self.avg = MinPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        elif(avgType == 4): #max before
          self.avg = MaxBeforePool1d(kernel_size=kernel_size, stride=stride, padding=0,seq_len=seq_len)
        elif(avgType == 5): #MIN before
          self.avg = MinBeforePool1d(kernel_size=kernel_size, stride=stride, padding=0,seq_len=seq_len)
        elif(avgType == 6): #avg before
          self.avg = AvgBeforePool1d(kernel_size=kernel_size, stride=stride, padding=0,seq_len=seq_len)


    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class MaxAvg(nn.Module):
    def __init__(self, kernel_size, stride, threshold=0.1):
        super(MaxAvg, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold

        # Max pooling layer
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=0)

        # Min pooling layer
        self.min_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

        self.ScalerParam = 0.5
        # self.ScalerParam = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        # Padding
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        # Combined max-min pooling
        max_pooled = self.max_pool(x.permute(0, 2, 1))
        # min_pooled = self.min_pool(x.permute(0, 2, 1))
        min_pooled = self.min_pool(x.permute(0, 2, 1))

        # Dynamically select between max and min pooling based on a condition
        # condition = (min_pooled - self.threshold > max_pooled).float()  # Use float to avoid NaN gradients
        # pooled = (1 - condition) * max_pooled + condition * min_pooled
        # max_pooled[abs(min_pooled)>abs(max_pooled)] = min_pooled[abs(min_pooled)>abs(max_pooled)]
        x = max_pooled * self.ScalerParam + (1-self.ScalerParam) * min_pooled
        x = x.permute(0,2,1)

        return x #pooled.permute(0, 2, 1)  # Reverse permutation

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size,avgType=1,seq_len=336):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1,avgType=avgType,seq_len=seq_len)
        if(avgType == 7):
          self.moving_avg = MaxAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        # moving_mean = self.moving_avg(moving_mean)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs,kernel_size=[25]):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.batch_size = configs.batch_size

        self.individual = configs.individual
        self.channels = configs.enc_in

        self.decompsition = series_decomp(25,2,self.seq_len)


        self.SelectorScaler = torch.nn.Parameter(torch.tensor([0.25]))
        self.SelectorScaler1 = torch.nn.Parameter(torch.tensor([0.5]))
        self.SelectorScaler2 = torch.nn.Parameter(torch.tensor([1.0]))

        self.ScalerParam1 = torch.nn.Parameter(torch.tensor([0.25]))
        self.ScalerParam2 =  torch.nn.Parameter(torch.tensor([0.25]))
        self.ScalerParam3 =  torch.nn.Parameter(torch.tensor([0.25]))
        self.ScalerParam4 = torch.nn.Parameter(torch.tensor([0.25]))

        self.TimeStampsSeq = nn.Sequential(
                  nn.Linear(self.seq_len,self.seq_len),
              )

        self.B_Linear =nn.Sequential(
                  nn.Linear(self.seq_len,self.pred_len),
              )
        self.F_Linear =nn.Sequential(
                  nn.Linear(self.seq_len*4,self.pred_len),
              )


    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark,y=None):
        # x: [Batch, Input length, Channel]


        # Normalisation and Scaling
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last

        z = x.detach()

        x = self.SelectorScaler * x


        # timestamps embedding
        timestamps = batch_x_mark
        timestamps = self.TimeStampsSeq(batch_x_mark.permute(0,2,1)).permute(0,2,1)

        timestamps = (timestamps[:,:,0] * self.ScalerParam1 +
                      timestamps[:,:,1] * self.ScalerParam2 +
                      timestamps[:,:,2] * self.ScalerParam3 +
                      timestamps[:,:,3] * self.ScalerParam4 ).unsqueeze(2)

        timestamps = timestamps.mean(axis=1).unsqueeze(1)



        # Multi Scale Decomposition
        R, T = self.decompsition(z)
        R_25, T_25 = self.decompsition(x)
        R_50, T_50 = self.decompsition(z*0.5)

        combined = torch.cat((R,R_50,T_25,x),dim=1)
        forecast = self.F_Linear(combined.permute(0,2,1)).permute(0,2,1)


        # Base Recover
        y = self.B_Linear(x.permute(0,2,1) * timestamps ).permute(0,2,1)/self.SelectorScaler


        y = y * forecast  + seq_last

        return y # to [Batch, Output length, Channel]


