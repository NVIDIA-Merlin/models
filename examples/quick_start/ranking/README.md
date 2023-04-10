# Quick-start for ranking models with Merlin

This is a template for building a pipeline for preprocessing, training and exporting ranking models for serving. It is composed by generic scripts that can be used with your own dataset. We also share some best practices for preprocessing and training ranking models.

In this example, we use the [TenRec dataset](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html), which is large (140 million positive interactions from 5 million users), contains explicit negative feedback (items exposed to the user and not interacted) and multiple target columns (click, like, share, follow).

You will learn later how to preprocess and train your own dataset.   
**TODO**: Add link to the section on how to use your own data

## Setup
You can run these scripts either using the latest Merlin TensorFlow image or installing the necessary Merlin libraries according to their documentation ([core](https://github.com/NVIDIA-Merlin/core), [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular), [dataloader](https://github.com/NVIDIA-Merlin/dataloader), [models](https://github.com/NVIDIA-Merlin/models/)).  
In this doc we provide the commands for setting up a Docker container for Merlin TensorFlow, as it provides all necessary libraries already installed.

### Download the TenRec dataset
You can find the TenRec dataset in this [link](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html). You might switch the page language to *English* in the top-left link, if you prefer.  
To be able to download the data, you need first to agree with the terms an register your e-mail. After downloading the zipped file (4.2 GB) you just need to uncompress the data.


## Preparing the data
The TenRec dataset contains a number of CSV files. We will be using the `QK-video.csv`, which logs user interactions with different videos.   

Here is an example on how the data looks like. For ranking models, you typically have user, item and contextual features and one or more targets, that can be a binary (e.g. has the customer clicked or liked and item) or regression target (e.g. watch times).

![TenRec dataset structure](../images/tenrec_dataset.png)

As `QK-video.csv` has a reasonable size (~15 GB), feel free to reduce it for less rows you want to test the pipeline more quickly or if you don't have a powerful GPU available (V100 with 32 GB or better). For example, with the following command you can truncate the file keeping the first 10 million rows (header line included).

```bash
head -n 10000001 QK-video.csv > QK-video-10M.csv
```

### Start Docker container

1. Pull the latest [Merlin TensorFlow image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow).  

```bash
docker pull nvcr.io/nvidia/merlin/merlin-tensorflow:latest 
```

2. Set `INPUT_DATA_PATH` variable to the folder where `QK-video.csv` was saved.  
The `OUTPUT_PATH` is the place where the preprocessed dataset and trained model will be saved.

```bash
INPUT_DATA_PATH=/path/to/input/dataset/
OUTPUT_PATH=/path/to/output/path/
```

3. Start a Merlin TensorFlow container in interactive mode
```bash
docker run --gpus all --rm -it --ipc=host -v $INPUT_DATA_PATH:/data -v $OUTPUT_PATH:/outputs \
  nvcr.io/nvidia/merlin/merlin-tensorflow:latest /bin/bash
```


## Preprocessing

In order to make it easy getting the data ready for model training, we provide a generic script: `preprocessing.py`. That script is based on **dask_cudf** and **NVTabular** libraries that leverage GPUs for accelerated and distributed preprocessing.  
P.s. **NVTabular** also supports CPU which is suitable for prototyping in dev environments.

The preprocessing script outputs preprocessed data as a number of parquet files, as well as a *schema* that stores output features metadata like statistics and tags.
**TODO**: Explain where schema is currently saved.

In this example, we set some options for preprocessing. Here is the explanation of the main arguments, and you can find the full documentation (here)[TODO].

- `--categorical_features` - Names of the categorical/discrete features (concatenated with "," without space).
- `--binary_classif_targets` - Names of the available target columns for binary classification task
- `--regression_targets` - Names of the available target columns for regression task
- `--user_id_feature` - Name of the user id feature
- `--item_id_feature` - Name of the item id feature
- `--to_int32`, `--to_int16`, `--to_int8` - Allows type casting the columns to the lower possible precision, which may avoid memory issues with large datasets.
- `--min_user_freq` - Removes examples of interactions from users with lower frequency than a threshold
- `--persist_intermediate_files` - Whether should persist/cache to disk intermediate files during preprocessing. This is useful for large datasets.
- `--dataset_split_strategy` - Strategy for splitting train and eval sets. In this case, `random_by_user` is chosen, which means that train and test will have the same users with some random examples reserved for evaluation.
- `--random_split_eval_perc` - Percentage of data to reserve for eval set
- `filter_query` - A filter query condition compatible with dask-cudf `DataFrame.query()`

```bash
cd /models/examples/quick_start/scripts/preproc/
OUT_DATASET_PATH=/outputs/
python preprocessing.py --input_data_format=csv --csv_na_values=\\N --input_data_path /data/QK-video.csv --filter_query="click==1 or (click==0 and follow==0 and like==0 and share==0)" --output_path=$OUT_DATASET_PATH --categorical_features=user_id,item_id,video_category,gender,age --binary_classif_targets=click,follow,like,share --regression_targets=watching_times --to_int32=user_id,item_id --to_int16=watching_times --to_int8=gender,age,video_category,click,follow,like,share --user_id_feature=user_id --item_id_feature=item_id --min_item_freq=30 --min_user_freq=30 --max_user_freq=150 --num_max_rounds_filtering=5 --dataset_split_strategy=random_by_user --random_split_eval_perc=0.2
```


After you execute this script, a folder `preproc` will be created in `--output_path` with the preprocessed datasets (with `train` and `eval` folders). You will find a number of partitioned parquet files in those dataset folders, as well as the `schema.yaml` file produced by `NVTabular` which is very important for automated model building in the next step.

## Training a ranking model
Merlin Models is a Merlin library that makes it easy to build and train RecSys models. It is built on top of TensorFlow, and provides building blocks for creating input layers based on the features in the schema, different feature interaction layers and output layers based on the target columns defined in the schema.

A number of popular ranking models are available in Merlin Models like **DLRM**, **DCN-v2**, **Wide&Deep**, **DeepFM**.

In the following generic `ranking_train_eval.py` script, you can easily train the popular **DLRM** model which performs 2nd level feature interaction. It sets `--model dlrm` and `--embeddings_dim 64` because DLRM models require all categorical columns to be embedded with the same dimension for the feature interaction. You notice that we can set many of the common model (e.g. `--mlp_layers`) and training hyperparameters like learning rate (`--lr`) and its decay (`--lr_decay_rate`, `--lr_decay_steps`), L2 regularization (`--l2_reg`, `embeddings_l2_reg`), `--dropout` among others.  We set `--epochs 1` and `--train_steps_per_epoch 10` to train for just 10 batches and make runtime faster. If you have a more GPU with more memory (e.g. V100 with 32 GB), you might increase `--train_batch_size` and `--eval_batch_size` to a much larger batch size, for example to `65536`.

### Dealing with class unbalance
There are many target columns available in the dataset, and you can select one of them for training by setting `--tasks=click`.  
In this dataset, there are about 3.7 negative examples (`click=0`) for each positive example (`click=1`). That leads to some class unbalance. We can couple with that by setting `--stl_positive_class_weight 4` to give more weight to the loss for positive examples, which are rarer

You can find the full documentation of the training script arguments [here](TODO).


```bash
cd /models/examples/quick_start/scripts/ranking/
CUDA_VISIBLE_DEVICES=0 TF_GPU_ALLOCATOR=cuda_malloc_async python  ranking_train_eval.py --train_path $OUT_DATASET_PATH/dataset/train --eval_path $OUT_DATASET_PATH/dataset/eval --output_path ./outputs/ --tasks=click --stl_positive_class_weight 3 --model dlrm --embeddings_dim 64 --l2_reg 1e-2 --embeddings_l2_reg 1e-6 --dropout 0.05 --mlp_layers 64,32  --lr 1e-4 --lr_decay_rate 0.99 --lr_decay_steps 100 --train_batch_size 65536 --eval_batch_size 65536 --epochs 1 
//--train_steps_per_epoch 10 
```


## Training a ranking model with multi-task learning
When multiple targets are available for the same features, models typically benefit from joint training a single model with multiple heads / losses. Merlin Models supports some architectures designed specifically for multi-task learning based on experts. You can find an example notebook with detailed explanations [here](https://github.com/NVIDIA-Merlin/models/blob/main/examples/usecases/ranking_with_multitask_learning.ipynb).


The `ranking_train_eval.py` script makes it easy to train ranking models with multi-task learning by setting more than one target, e.g. `--tasks="click,like,follow,share"`). 


### Training an MMOE model
In the following example, we use the popular **MMOE** (`--model mmoe`) architecture for multi-task learning. It creates independent expert sub-networks (as defined by `--mmoe_num_mlp_experts`, `--expert_mlp_layers`) that interacts independently the input features. Each task has a gate with `--gate_dim` that averages the expert outputs based on learned softmax weights, so that each task can harvest relevant information for its predictions. Each task might also have an independent tower by setting `--tower_layers`.  
You can also balance the loss weights by setting `--mtl_loss_weight_*` arguments and the tasks positive class weight by setting `--mtl_pos_class_weight_*`.

```bash
cd /models/examples/quick_start/scripts/ranking/
DATA_PATH=/quick_start/scripts/preproc/output/final_dataset

CUDA_VISIBLE_DEVICES=0 TF_MEMORY_ALLOCATION=0.8 python  ranking_train_eval.py --train_path $DATA_PATH/train --eval_path $DATA_PATH/eval --output_path ./outputs/ --tasks=click,like,follow,share --model mmoe --mmoe_num_mlp_experts 3 --expert_mlp_layers 128 --gate_dim 32 --use_task_towers --tower_layers 64 --embedding_sizes_multiplier 4 --l2_reg 1e-5 --embeddings_l2_reg 1e-6 --dropout 0.05  --lr 1e-4 --lr_decay_rate 0.99 --lr_decay_steps 100 --train_batch_size 65536 --eval_batch_size 65536 --epochs 1 --mtl_pos_class_weight_click=1 --mtl_pos_class_weight_follow=1 --mtl_pos_class_weight_like=1 --mtl_loss_weight_click=4 --mtl_loss_weight_follow=3 --mtl_loss_weight_like=2 --mtl_loss_weight_share=1 
//--train_steps_per_epoch 3  
```

## TODO
- Refine this overview documentation and command examples
- Create section to teach how to use the generic scripts with user's own data (`using_your_data.md`)
- Create another document explaining all arguments for `preprocessing.py` and `ranking_train_eval.py`
- Create another document providing best practices on setting hyperparameters for ranking models based on the empirical results from our research experimentation (e.g. hparam optimization search space, best hparams found, comparison of the accuracy of STL and MTL models for each task) - `best_practices.md`
- Refine scripts to accept both CLI args or YAML args