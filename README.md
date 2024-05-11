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

### Model Architecture

![model architecture](https://raw.githubusercontent.com/seyf1elislam/UMS-Linear/main/imgs/ums-linear_architecture.png)

The model is based on an MLP-based model that uses timestamps and multiscale decomposition to boost the performance of MLP-based models. The model follows these steps:

1. **Timestamp Embedding**: The model uses timestamps as input and embeds them using a simple dense layer.
2. **Multiscale Decomposition**: After normalizing the data, the model scales the data into multiple scales using the following scales: [100%, 50%, 25%]. It then decomposes the data using trend and remainder decomposition. The decomposed data is passed to the MLP model. This helps the model reduce the effects of noise and focus on data trends.
3. **Residual Connection**: The model uses residual connections to connect the decomposed data with the original data. This helps the model follow data trends and sharp changes.
4. **Reconstruction Phase**: By multiplying the results of each layer, the model reconstructs the final output.

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

## Loss Function:

> In this paper, we used a combination of MSE and MAE with diff ($diff=X_{i}- X_{i-1}$): 0.33 × MSE + 0.33 × MAE + 0.33 × diff <br/> `Note: This loss function is used only during training. During evaluation, we used pure MSE and MAE without any modifications.`

## Model Evaluation

During evaluation, we used MSE and MAE to compare the results with other existing models.

## Results

![model evaluation](https://raw.githubusercontent.com/seyf1elislam/UMS-Linear/main/imgs/ums-linear-results.png)

The results show the dominance of UMS-Linear over the base models Nlinear, Dlinear, and other models such as PatchTST, TimesNet, FedFormer, AutoFormer, and others on univariate benchmarks ETTh1, ETTh2, ETTm1, ETTm2, and all horizons (96, 192, 336, 720).

## Visualization:

> Dlinear vs UMS-Linear <br/> ![dlinear vs umslinear](https://raw.githubusercontent.com/seyf1elislam/UMS-Linear/main/imgs/umslinear_vs_dlinear_plot.png)

## Datasets

- [https://github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)

## Acknowledgement

> - Thanks to the authors of the original LSTF-Linear model and the authors of the AutoFormer paper for sharing their model implementations.
