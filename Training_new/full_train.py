#@title  # $\color{cyan}{\text{full train func  (#3)}}$
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from Training_new.train_it import train_it
# %matplotlib inline

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def getModelDict(model_class):
  return AttributeDict({"Model":model_class})

def full_train(
    args,
    global_model_dict,
    model="Linear",
    custom_model=None,
    features="S",
    train_epochs=20,
    enc_in=1,
    seq_len=336,pred_lens=[96,192,336,720],
    data="ETTh1",learning_rate = 0.005,batch_size = 32,seed=2021,
    add_linear_to_name = False,
    note="",
    use_edited_exp = True,
    use_custom_loss=False,
    # custom_loss =default_loss,
    custom_loss =lambda batch_y, outputs, criterion: criterion(outputs, batch_y),
    save_return_dict_asfile=False,
    patience =3,
    ):
  torch.cuda.empty_cache()
  fix_seed = seed
  random.seed(fix_seed)
  torch.manual_seed(fix_seed)
  np.random.seed(fix_seed)

  # SPLinear = getModelDict(Patch_Mixer)
  if not custom_model is None :
    CurrentModel = getModelDict(custom_model)
    if add_linear_to_name :
      model = f"{model}_Linear"
    global_model_dict[model] =CurrentModel
  #=========================

  # args.timeenc = 0
  args.data_path=f"{data}.csv"
  args.data=data
  args.learning_rate = learning_rate
  args.train_epochs = train_epochs
  args.model=model
  args.patience=patience
  # args.model_id=f"ETTh1_{seq_len}_{pred_len}"

  args.features = features
  args.batch_size=batch_size
  args.enc_in = enc_in
  args.individual = False

  # args.patch_len = 16
  # args.stride = 8
  # args.padding_patch = "end"
  #=========================

  args.seq_len=seq_len
  results = []
  for pred_len in pred_lens:
    args.model_id=f"{data}_{seq_len}_{pred_len}_"
    args.pred_len=pred_len
    mse, mae,settings = train_it(
      args,global_model_dict,
      use_edited_exp=use_edited_exp ,
      custom_loss=custom_loss,
      fix_seed=seed,
      save_return_dict_asfile=save_return_dict_asfile,
    )
    results.append({
        "pred_len":pred_len,
        "mse":mse,
        "mae":mae,
        "experiment":f"[model={model}_dataset={data}_lookback={seq_len}_pred={pred_len}_seed={seed}_lr={learning_rate}_batchsize={batch_size}]",
    })
  return {
      "settings":settings,
      "model":model,
      "note":note,
      "results":results,
  }
