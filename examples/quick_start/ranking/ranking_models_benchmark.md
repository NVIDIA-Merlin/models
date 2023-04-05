# Benchmark of ranking models
In this document, we describe a benchmark of ranking models available in Merlin Models library for the TenRec dataset, which you can use as a reference when deciding the models an hyperparameters you might want to try with your own dataset. We also share the search space used for hyperparameter optimization, that you reuse for your own hyperparameter tuning.
These results are reproduceable by using the TenRec and the quick-start scripts for preprocessing and training. 

## Neural ranking models.
This benchmark includes the following neural architectures for ranking, with the corresponding paper references:

Single-Task Learning (STL)
- MLP - Simple multi-layer perceptron architecture. 
- Wide&Deep (paper) (`MLPBlock`) - ....
- DeepFM (paper) (`DeepFMModel`) - ...
- DLRM (paper) (`DLRMModel`) - ...
- DCN-v2 (paper) (`DCNModel`) - ...

Multi-Task Learning (MTL)
- MLP (paper) (`MLPBlock`) - ...
- MMOE (paper) (`MMOEBlock`) - ...
- PLE (paper)(`PLEBlock`)  - ...


## Hyperparameter tuning setup
We ran a separate hyperparameter tuning process for each model using TenRec dataset, which we call experiment group. We use the Weights&Biases Sweeps feature for managing the hypertuning process for each experiment group. The hypertuning uses bayesian optimization and we use as maximization objective the average AUC of the 
We use the following strategy for for hypertuning the models:

- **Single-task learning**: All 

- Search space
- Plot that shows accuracy improving over steps
- Most important hparams per model type
- Benchmark of accuracy and runtime
