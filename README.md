# Univariate Multiscale Decomposition: Improving Linear Models for Univariate Long-term Time Series Forecasting 

This repo is the official PyTorch implementation of UMS-Linear [Univariate Multiscale Decomposition: Improving Linear Models for Univariate Long-term Time Series Forecasting)](https://github.com/seyf1elislam/UMS-Linear/). The model is implemented in PyTorch and tested on multiple benchmark datasets. It outperforms the existing state-of-the-art models on multiple univariate time series forecasting benchmarks.

## Updates

- [x] UMS-Model added (2024-02-05)

## Detailed Description

| Files                                                     | Description                                                                                                     |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `UMS-Linear.ipynb`                                        | Jupyter notebook for testing UMS-Linear                                                                         |
| `Models/UMS-Linear.py`                                    | Contains the implementation of UMS-Linear                                                                       |
| `/Training_new/train_it.py` `/Training_new/full_train.py` | contains the Training Function that runs experiments                                                            |
| `/exp/exp_main_edit1.py`                                  | Contains the Exp class for UMS-Linear compatible with `/Training_new/train_it.py` `/Training_new/full_train.py` |

The rest of the files are the same as the original container used by LTSF-Linear and AutoFormer. It contains the data provider and the Exp class. We have added a custom Exp class for UMS-Linear, providing more customization for training.

## UMS-Linear Model

This simple architecture gives UMS-Linear the following characteristics:

> - High efficiency in terms of the number of parameters and computation
> - High speed in terms of training and inference
> - High accuracy in terms of univariate forecasting
> - O(1) time/memory complexity
> - Small number of parameters that can be trained with the lowest computational cost

### Impact of Timestamps:

> The impact of timestamps is visualized by comparing a simple linear model with a linear model with timestamps.<br/> ![alt text](https://raw.githubusercontent.com/seyf1elislam/UMS-Linear/main/imgs/linear_vs_timestamp_linear_plot.png)

## Model Training

The model was trained on a Free Tier Colab, using the Adam optimizer, for less than 15 epochs with an early stopping patience factor of 3, a batch size of 336 or 512, and a learning rate of 0.001 or 0.0005. The model was trained on multiple benchmark datasets and outperformed the existing state-of-the-art models on multiple univariate benchmarks.


## Model Evaluation

During evaluation, we used MSE and MAE to compare the results with other existing models.

## Results

The results show the dominance of UMS-Linear over the base models Nlinear, Dlinear, and other models such as PatchTST, TimesNet, FedFormer, AutoFormer, and others on univariate benchmarks ETTh1, ETTh2, ETTm1, ETTm2, and all horizons (96, 192, 336, 720).

## Visualization:

> Dlinear vs UMS-Linear <br/> ![dlinear vs umslinear](https://raw.githubusercontent.com/seyf1elislam/UMS-Linear/main/imgs/umslinear_vs_dlinear_plot.png)

## Datasets

- [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)

## Acknowledgement

> - Thanks to the authors of the original LSTF-Linear model and the authors of the AutoFormer paper for sharing their model implementations.
