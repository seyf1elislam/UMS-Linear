import torch
import random
import numpy as np
from exp.exp_main import Exp_Main
from exp.exp_main_edit1 import Exp_Main_Edit1


def train_it(args,global_model_dict,use_edited_exp=True,
             use_custom_loss=False,
             custom_loss=None,
             fix_seed=None,
             save_return_dict_asfile=False,use_visual=True):
  #=========================
  args.use_visual = use_visual
  if use_edited_exp:
    Exp = Exp_Main_Edit1
  else :
    Exp = Exp_Main
  if fix_seed is None:
    fix_seed = random.randint(0, 100000)
  #=========================
  random.seed(fix_seed)
  torch.manual_seed(fix_seed)
  np.random.seed(fix_seed)
  torch.cuda.manual_seed(fix_seed)
  if args.is_training:
      for ii in range(args.itr):
          # setting record of experiments
          setting = f"{args.model_id}{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}eb{args.embed}dt{args.distil}{args.des}{ii}"
          if use_edited_exp:
            exp = Exp(args,global_model_dict=global_model_dict)  # set experiments
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            # exp.train(setting , use_custom_loss=use_custom_loss)
            exp.train(setting , custom_loss=custom_loss)
          else :
            if custom_loss :
               print("please note that custom_loss linked only with edited main exp")
            exp = Exp(args)  # set experiments
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

          if not args.train_only:
              print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
              exp.test(setting, save_return_dict_asfile=save_return_dict_asfile)
              mse, mae = exp.test(setting)
          if args.do_predict:
              print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
              exp.predict(setting, True)

          torch.cuda.empty_cache()
      return mse, mae , setting
  else:
      ii = 0
      # setting = f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}'
      setting = f"{args.model_id}{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}eb{args.embed}dt{args.distil}{args.des}{ii}"
      exp = Exp(args)  # set experiments

      if args.do_predict:
          print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
          exp.predict(setting, True)
      else:
          print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
          mse, mae = exp.test(setting, test=1)
      torch.cuda.empty_cache()
      return mse, mae , setting
