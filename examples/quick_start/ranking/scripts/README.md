# Quick-start for ranking models with Merlin

This is a template for building a pipeline for preprocessing, training and exporting ranking models. It is composed by generic scripts that can be used with your own dataset. We also share some best practices for preprocessing and training ranking models.

In this example, we use the TenRec dataset from Alibaba, which has a reasonable size (Xrows), contains explicit negative interactions (items not interacted by users) and multiple target columns (click, like, share, follow).

## Setup
You can run these scripts either using the last Merlin TensorFlow image or installing the necessary Merlin libraries according to their documentation (core, NVTabular, dataloader, models).
In this doc we provide the commands for setting up a Merlin TensorFlow, as it provides all necessary libraries already installed.

### Download the TenRec dataset
In this step you [download](https://static.qblv.qq.com/qblv/h5/algo-frontend/tenrec_dataset.html) and uncompress...

You will learn later how to preprocess and train your own dataset. 

## Preparing the data
The TenRec dataset contains a number of CSV files. We will be using the `QK-video.csv`, which logs user interactions with different videos. As the dataset has a reasonable size (~15 GB), feel free to reduce that file for less rows you want to test the pipeline more quickly or don't have a powerful GPU available (V100 with 32 GB or better). For example, the following command you can truncate the file, keeping the first 10 million rows (header line included).

```bash
head -n 10000001 QK-video.csv > QK-video-10M.csv
```

### Start Docker container

1. Pull the latest [Merlin TensorFlow image](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow).  

```bash
docker pull nvcr.io/nvidia/merlin/merlin-tensorflow:latest 
```

2. Set `INPUT_DATA_PATH` variable to the folder where TenRec dataset is saved. The `OUTPUT_PATH` is the place where the preprocessed dataset and trained model will be saved.

```bash
INPUT_DATA_PATH=/path/to/input/dataset/
OUTPUT_PATH=/path/to/output/path/
```

3. Start a Merlin TensorFlow container in iterative mode
```bash
docker run --gpus all --rm -it --ipc=host -v $INPUT_DATA_PATH:/data -v $OUTPUT_PATH:/outputs \
  nvcr.io/nvidia/merlin/merlin-tensorflow:latest /bin/bash
```


## Preprocessing

In order to make it easy getting the data ready for model training, we provide a generic script: `preprocessing.py`. The script is based on **dask_cudf** and **NVTabular** libraries, that leverages GPUs for preprocessing (and scales to multiple GPU). It outputs a number of parquet files with the preprocessed data, as well as a `schema.yaml` file that stores features metadata like statistics and tags.

In this example, we set some options for preprocessing. Here is the explanation of the main arguments, and you can find the full documentation (here)[TODO].

- `--categorical_features` - Names of the categorical/discrete features (concatenated with "," without space).
- `--binary_classif_targets` - Names of the available target columns for binary classification task
- `--regression_targets` - Names of the available target columns for regression task
- `--user_id_feature` - Name of the user id feature
- `--item_id_feature` - Name of the item id feature
- `--to_int32`, `--to_int16`, `--to_int8` - Allows type casting the columns to the lower possible precision, which may avoid memory issues with large datasets.
- `--min_user_freq` - Removes examples of interactions from users with lower frequency than a threshold
- `--persist_intermediate_files` - Whether should persist to disk intermediate files during preprocessing. This is useful for large datasets.
- `--dataset_split_strategy` - Strategy for splitting train and eval sets
- `--random_split_eval_perc` - Percentage of data to reserve for eval set

```bash
cd /examples/quick_start/ranking/scripts/preproc/
OUT_DATASET_PATH=/outputs/preproc/
python preprocessing.py --input_data_format=csv --csv_na_values=\\N --input_data_path /data/QK-video-10M.csv --output_path=/outputs/preproc/ --categorical_features=user_id,item_id,video_category,gender,age --binary_classif_targets=click,follow,like,share --regression_targets=watching_times --to_int32=user_id,item_id --to_int16=watching_times --to_int8=gender,age,video_category,click,follow,like,share --user_id_feature=user_id --item_id_feature=item_id --min_user_freq 5 --persist_intermediate_files --dataset_split_strategy=random --random_split_eval_perc=0.2 	
```

After you execute this script, a folder `preproc` will be created in `--output_path` with the preprocessed datasets (with `train` and `eval` folders). You will find a number of partitioned parquet files in those dataset folders, as well as the `schema.yaml` file produced by `NVTabular` which is very important for automated model building in the next step.

## Training a ranking model
Merlin Models is a Merlin library that makes it easy to build and train RecSys models. It is built on top of TensorFlow, and provides building blocks for creating input layers based on the features in the schema, different feature interaction layers and output layers based on the target columns defined in the schema.

A number of popular ranking models are available in Merlin Models like **DLRM**, **DCN-v2**, **Wide&Deep**, **DeepFM**.

In the following generic `ranking_train_eval.py` script, you can easily train the popular **DLRM** model which performs 2nd level feature interaction. It sets `--model dlrm` and `--embeddings_dim 64`. because DLRM models require all categorical columns to be embedded with the same dimension for the feature interaction. You notice that we can set many of the common model (e.g. `--mlp_layers`) and training hyperparameters like learning rate (`--lr`) and its decay (`--lr_decay_rate`, `--lr_decay_steps`), L2 regularization (`--l2_reg`, `embeddings_l2_reg`), `--dropout` among others.  We set `--epochs 1` and `--train_steps_per_epoch 10` to train for just 10 batches and make runtime faster. If you have a more GPU with more memory (e.g. V100 with 32 GB), you might increase `--train_batch_size` and `--eval_batch_size` to a much larger batch size, for example 65536.

### Dealing with class unbalance
As there are many target columns available in the dataset, you can select only one of them for training by setting `--tasks=click`.  
In this dataset, there are about 3.7 negative examples (`click=0`) for each positive example (`click=1`). That leads to some class unbalance. We can couple with that by setting `--stl_positive_class_weight 4` to give more weight to the loss for positive examples, which are rarer

You can find the full documentation of the training script arguments [here](TODO).


```bash
cd /examples/quick_start/ranking/scripts/training/
CUDA_VISIBLE_DEVICES=0 TF_MEMORY_ALLOCATION=0.8 python  ranking_train_eval.py --train_path $OUT_DATASET_PATH/final_dataset/train --eval_path $OUT_DATASET_PATH/final_dataset/eval --output_path ./outputs/ --tasks=click --stl_positive_class_weight 4 --model dlrm --embeddings_dim 64 --l2_reg 1e-5 --embeddings_l2_reg 1e-6 --dropout 0.05 --mlp_layers 64,32  --lr 1e-4 --lr_decay_rate 0.99 --lr_decay_steps 100 --train_batch_size 4096 --eval_batch_size 4096 --epochs 1 --train_steps_per_epoch 10 
```


## Training a ranking model with multi-task learning
When multiple targets are available for the same features, models typically benefit from joint training a single model with multiple heads / losses. Merlin Models supports some architectures designed specifically for multi-task learning based on experts. You can find an example notebook with detailed explanations [here](https://github.com/NVIDIA-Merlin/models/blob/main/examples/usecases/ranking_with_multitask_learning.ipynb).


The `ranking_train_eval.py` script makes it easy to train ranking models with multi-task learning by setting more than one target in `--tasks` (e.g. `click,like,follow,share`). 

### Setting tasks sample space
In that dataset, some targets depend on others. For example, you only have a `like/follow/share=1` event if the user has clicked in the item. The learning of the dependent tasks is better if we set the appropriate sample space for the targets. In this case, we want to train the `click` target  using the entire space, and train the other targets (i.e., compute the loss) only for click space (where `click=1`).  

The scripts allows for setting the tasks sample space by using `--tasks_sample_space`, where the position should match the order of the `--tasks`. Empty value means the task will be trained in the entire space, i.e., loss computed for all examples in the dataset.


### Training an MMOE model
In the following example, we use the popular **MMOE** (`--model mmoe`) architecture for multi-task learning. It creates independent expert sub-networks (as defined by `--mmoe_num_mlp_experts`, `--expert_mlp_layers`) that interacts independently the input features. Each task has a gate with `--gate_dim` that averages the expert outputs based on learned softmax weights, so that each task can harvest relevant information for its predictions. Each task might also have an independent tower between the gates output and the last layer (binary prediction) by setting `--tower_layers`.  
You can also balance the loss weights by setting `--mtl_loss_weight_*` arguments and the tasks positive class weight by setting `--mtl_pos_class_weight_*`.

```bash
cd /quick_start/scripts/ranking/
DATA_PATH=/quick_start/scripts/preproc/output/final_dataset

CUDA_VISIBLE_DEVICES=0 TF_MEMORY_ALLOCATION=0.8 python  ranking_train_eval.py --train_path $DATA_PATH/train --eval_path $DATA_PATH/eval --output_path ./outputs/ --tasks=click,like,follow,share --tasks_sample_space=,click,click,click --model mmoe --mmoe_num_mlp_experts 3 --expert_mlp_layers 128 --gate_dim 32 --tower_layers 64 --embedding_sizes_multiplier 4 --l2_reg 1e-5 --embeddings_l2_reg 1e-6 --dropout 0.05  --lr 1e-4 --lr_decay_rate 0.99 --lr_decay_steps 100 --train_batch_size 4096 --eval_batch_size 65536 --epochs 1 --mtl_pos_class_weight_click=1 --mtl_pos_class_weight_follow=1 --mtl_pos_class_weight_like=1 --mtl_loss_weight_click=4 --mtl_loss_weight_follow=3 --mtl_loss_weight_like=2 --mtl_loss_weight_share=1 --use_task_towers --train_steps_per_epoch 3 --in_batch_negatives_train 0 
```

## TODO
- Refine this overview documentation and command examples
- Create section to teach how to use the generic scripts with user's own data
- Create another document explaining all arguments for `preprocessing.py` and `ranking_train_eval.py`
- Create another document providing best practices on setting hyperparameters for ranking models based on the empirical results from our research experimentation (e.g. hparam optimization search space, best hparams found, comparison of the accuracy of STL and MTL models for each task)
- Refine scripts to accept both CLI args or YAML args
- Refine `preprocessing.py` to provide additional dataset split strategies (e.g. `random_by_user`, `temporal`).
- Test `preprocessing.py` with larger/full dataset