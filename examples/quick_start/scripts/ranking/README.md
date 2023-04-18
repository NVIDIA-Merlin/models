# Ranking script
The `ranking.py` is a template script that leverages the Merlin [models](https://github.com/NVIDIA-Merlin/models/) library (Tensorflow API) to build, train, evaluate ranking models. In the end you can either save the model for interence or persist model predictions to file.

Merlin models provide building blocks on top of Tensorflow (Keras) that make it easy to build and train advanced neural ranking models. There are blocks for representing input, configuring model outputs/heads, perform feature interactions, losses, metrics, negative sampling, among others.

## Ranking in multi-stage RecSys
Large online services like social media, streaming, e-commerce, and news provide a very broad catalog of items and leverage recommender systems to help users find relevant items. Those companies typically deploy recommender systems pipelines with [multiple stages](https://medium.com/nvidia-merlin/recommender-systems-not-just-recommender-models-485c161c755e), in particular the retrieval and ranking. The retrieval stage selects a few hundreds or thousands of items from a large catalog. It can be a heuristic approach (like most recent items) or a scalable model like Matrix Factorization, [Two-Tower architecture](https://medium.com/nvidia-merlin/scale-faster-with-less-code-using-two-tower-with-merlin-c16f32aafa9f) or [YouTubeDNN](https://static.googleusercontent.com/media/research.google.com/pt-BR//pubs/archive/45530.pdf). Then, the ranking stage scores the relevance of the candidate items provided by the previous stage for a given user and context.


## Multi-task learning for ranking models
It is common to find scenarios where you need to score the likelihood of different user-item events, e.g., clicking, liking, sharing, commenting, following the author, etc. Multi-Task Learning (MTL) techniques have been popular in deep learning to train a single model that is able to predict multiple targets at the same time.

By using MTL, it is typically possible to improve the tasks accuracy for somewhat correlated tasks, in particular for sparser targets, for which less training data is available. And instead of spending computational resources to train and deploy different models for each task, you can train and deploy a single MTL model that is able to predict multiple targets.

You can find more details in this [post](https://medium.com/nvidia-merlin/building-ranking-models-powered-by-multi-task-learning-with-merlin-and-tensorflow-4b4f993f7cc3) on the multi-task learning building blocks provided by [models](https://github.com/NVIDIA-Merlin/models/)  library.

The `ranking.py` script makes it easy to use multi-task learning backed by models library. It is automatically enabled when you provide more than one target column to `--target` arguments.

## Supported models
The `ranking.py` script makes it easy to use baseline and advanced deep ranking models available in [models](https://github.com/NVIDIA-Merlin/models/) library.  
The script can be also used as an **advanced example** that demonstrate [how to set specific hyperparamters using models API](ranking_models.py).

### Baseline ranking architectures
- MLP (`MLPBlock`) - Simple multi-layer perceptron architecture. 
- Wide and Deep ([paper](https://dl.acm.org/doi/10.1145/2988450.2988454)) - ...
- DeepFM ([paper](https://arxiv.org/abs/1703.04247)) (`DeepFMModel`) - ...
- DRLM ([paper](https://arxiv.org/abs/1906.00091)) (`DLRMModel`) - ...
- DCN-v2 ([paper](https://dl.acm.org/doi/10.1145/3442381.3450078)) (`DCNModel`) - ...

### Multi-task learning architectures
- MMOE ([paper](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)) (`MMOEBlock`) - The Multi-gate Mixture-of-Experts (MMoE) architecture  was introduced by Google in 2018 and is one of the most popular models for multi-task learning on tabular data. It allows parameters to be automatically allocated to capture either shared task information or task-specific information. The core components of MMoE are experts and gates. Instead of using a shared-bottom for all tasks, it has multiple expert sub-networks processing input features independently from each other. Each task has an independent gate, which dynamically selects based on the inputs the level with which the task wants to leverage the output of each expert. The gate is typically just a small MLP sub-network that provides softmax scores over the number of experts given the inputs. Those scores are used as weights for computing a weighted average of the experts’ outputs and form an independent representation for each task.
- CGC ([paper](https://dl.acm.org/doi/10.1145/3383313.3412236)) (`CGCBlock`) - Instead of having tasks sharing all the experts like in MMOE, it propos allowing for splitting task-specific experts and shared experts, in an architecture named Customized Gate Control (CGC) Model.
- PLE ([paper](https://dl.acm.org/doi/10.1145/3383313.3412236)) (`PLEBlock`) - In the same paper introducing CGC, authors proposed stacking multiple CGC models on top of each other to form a multi-level MTL model, so that the model can progressively combine shared and task-specific experts. They name this approach as Progressive Layered Extraction (PLE). Their paper experiments showed accuracy improvements by using PLE compared to CGC.

<center>
<img src="https://miro.medium.com/v2/resize:fit:720/0*Fo6rIr10IJQCB6sb" alt="Multi-task learning architectures" >
</center>


## Best practices


Neural networks typically use embeddings (1D continuous vectors) to represent categorical features as input. The embeddings are stored in embedding layers or tables, whose first dim in the cardinality of the categorical feature and 2nd dim is the embedding size. In order to minimize the memory requirements of the embedding table, **the categorical values need to be encoded into contiguous ids in the preprocessing**, which will define the size of the embedding table in the model.

- stl_positive_class_weight
- Negative sampling: in_batch_negatives_train

Different ranking model characteristics
- STL: "mlp", "dcn", "dlrm", "deepfm", "wide_n_deep"
- MTL: "mmoe", "cgc","ple",

### Setting tasks sample space
In that dataset, some targets depend on others. For example, you only have a `like/follow/share=1` event if the user has clicked in the item. The learning of the dependent tasks is better if we set the appropriate sample space for the targets. In this case, we want to train the `click` target  using the entire space, and train the other targets (i.e., compute the loss) only for click space (where `click=1`).  

The scripts allows for setting the tasks sample space by using `--tasks_sample_space`, where the position should match the order of the `--tasks`. Empty value means the task will be trained in the entire space, i.e., loss computed for all examples in the dataset.

## Command line arguments

### Inputs

```
  --train_path
                        Path of the train set.
  --eval_path
                        Path of the eval set.
```

### Tasks
```
  --tasks               Columns (comma-sep) with the target columns to
                        be predicted. A regression/binary classification
                        head is created for each of the target columns.
                        If more than one column is provided, then multi-
                        task learning is used to combine the tasks
                        losses. If 'all' is provided, all columns tagged
                        as target in the schema are used as tasks. By
                        default 'all'
  --tasks_sample_space 
                        Columns (comma-sep) to be used as sample space
                        for each task. This list of columns should match
                        the order of columns in --tasks. Typically this
                        is used to explicitly model that the task event
                        (e.g. purchase) can only occur when another
                        binary event has already happened (e.g. click).
                        Then by setting for example
                        --tasks=click,purchase
                        --tasks_sample_space,click, you configure the
                        training to compute the purchase loss only for
                        examples with click=1, making the purchase
                        target less sparser.
```

### Model
```
  --model {mmoe,cgc,ple,dcn,dlrm,mlp,wide_n_deep,deepfm}
                        Types of ranking model architectures that are
                        supported. Any of these models can be used with
                        multi-task learning (MTL). But these three are
                        specific to MTL: 'mmoe', 'cgc' and 'ple'. By default
                        'mlp'
  --activation 
                        Activation function supported by Keras, like:
                        'relu', 'selu', 'elu', 'tanh', 'sigmoid'. By
                        default 'relu'
  --mlp_init            Keras initializer for MLP layers
  --l2_reg              L2 regularization factor. By default 1e-5.
  --embeddings_l2_reg 
                        L2 regularization factor for embedding tables.
                        It operates only on the embeddings in the
                        current batch, not on the whole embedding table.
                        By default 0.0
  --embedding_sizes_multiplier 
                        When --embedding_dim is not provided it infers
                        automatically the embedding dimensions from the
                        categorical features cardinality. This factor
                        allows to increase/decrease the embedding dim
                        based on the cardinality. Typical values range
                        between 2 and 10. By default 2.0
  --dropout             Dropout rate. By default 0.0
  --mlp_layers 
                        The dims of MLP layers. By default '128,64,32'
  --stl_positive_class_weight 
                        Positive class weight for single-task models. By
                        default 1.0. The negative class weight is fixed
                        to 1.0
```

### DCN-v2
```
  --dcn_interacted_layer_num
                        Number of interaction layers for DCN-v2
                        architecture. By default 1.
```

### DLRM and DeepFM
```
  --embeddings_dim 
                        Sets the embedding dim for all embedding columns
                        to be the same. This is only used for --model
                        'dlrm' and 'deepfm'
```

### Wide&Deep
```
  --wnd_hashed_cross_num_bins 
                        Used with Wide&Deep model. Sets the number of
                        bins for hashing feature interactions. By
                        default 10000.
  --wnd_wide_l2_reg 
                        Used with Wide&Deep model. Sets the L2 reg of
                        the wide/linear sub-network. By default 1e-5.
  --wnd_ignore_combinations 
                        Feature interactions to ignore. Separate feature
                        combinations with ',' and columns with ':'. For
                        example: --wnd_ignore_combinations='item_id:item
                        _category,user_id:user_gender'
```

### Wide&Deep and DeepFM
```            
  --multihot_max_seq_length
                        DeepFM and Wide&Deep support multi-hot
                        categorical features for the wide/linear sub-
                        network. But they require setting the maximum
                        list length, i.e., number of different multi-hot
                        values that can exist in a given example. By
                        default 5.
```

### MMOE
```
  --mmoe_num_mlp_experts 
                        Number of experts for MMOE. All of them are
                        shared by all the tasks. By default 4.
```

### CGC and PLE
```                        
  --cgc_num_task_experts 
                        Number of task-specific experts for CGC and PLE.
                        By default 1.
  --cgc_num_shared_experts 
                        Number of shared experts for CGC and PLE. By
                        default 2.
  --ple_num_layers 
                        Number of CGC modules to stack for PLE
                        architecture. By default 1.
```        

### Expert-based MTL models
```
  --expert_mlp_layers 
                        For MTL models (MMOE, CGC, PLE) sets the MLP
                        architecture of experts. By default '64'
  --gate_dim            Dimension of the gate dim MLP layer. By default
                        64
  --mtl_gates_softmax_temperature 
                        Sets the softmax temperature for the gates
                        output layer, that provides weights for the
                        weighted average of experts outputs. By default
                        1.0
```

### Multi-task learning models
```
  --use_task_towers 
                        Enables task-specific towers before its head.
  --tower_layers 
                        MLP architecture of task-specific towers. By
                        default '64'
```

### Negative sampling
```
  --in_batch_negatives_train 
                        If greater than 0, enables in-batch sampling,
                        providing this number of negative samples per
                        positive. This requires that your data contains
                        only positive examples, and that item features
                        are tagged accordingly in the schema, for
                        example, by setting --item_features in the
                        preprocessing script.
  --in_batch_negatives_eval 
                        Same as --in_batch_negatives_train for
                        evaluation.
```

### Training and evaluation
```
  --lr LR               Learning rate
  --lr_decay_rate 
                        Learning rate decay factor. By default 0.99
  --lr_decay_steps 
                        Learning rate decay steps. It decreases the LR
                        at this frequency, by default each 100 steps
  --train_batch_size 
                        Train batch size. By default 1024. Larger batch
                        sizes are recommended for better performance.
  --eval_batch_size 
                        Eval batch size. By default 1024. Larger batch
                        sizes are recommended for better performance.
  --epochs EPOCHS       Number of epochs. By default 1.
  --optimizer {adagrad,adam}
                        Optimizer. By default 'adam'
  --train_metrics_steps 
                        How often should train metrics be computed
                        during training. You might increase this number
                        to reduce the frequency and increase a bit the
                        training throughput. By default 10.
  --validation_steps 
                        If not predicting, logs the validation metrics
                        for this number of steps at the end of each
                        training epoch. By default 10.
  --random_seed 
                        Random seed for some reproducibility. By default
                        42.
  --train_steps_per_epoch 
                        Number of train steps per epoch. Set this for
                        quick debugging.
```

### Logging
```
  --metrics_log_frequency 
                        --How often metrics should be logged to
                        Tensorboard or Weights&Biases. By default each
                        50 steps.
  --log_to_tensorboard 
                        Enables logging to Tensorboard.
  --log_to_wandb 
                        Enables logging to Weights&Biases. This requires
                        sign-up for a free W&B account and providing an
                        API key in the console.
  --wandb_project 
                        Name of the Weights&Biases project to log
  --wandb_entity 
                        Name of the Weights&Biases team/org to log
  --wandb_exp_group 
                        Not used by the script. Just used to allow for
                        logging some info to organize experiments in
                        Weights&Biases
```

### Outputs
```
  --output_path 
                        Output path for saving predictions.
  --save_trained_model_path 
                        If provided, model is saved to this path after
                        training.
  --predict             If enabled, the dataset provided in
                        --eval_pathwill be used for prediction (instead
                        of evaluation). The prediction scores for the
                        that dataset will be saved to
                        --predict_output_path (or to --output_path),
                        according to the --predict_output_format choice.
  --predict_keep_cols 
                        Comma-separated list of columns to keep in the
                        output prediction file. If no columns is
                        provided, all columns are kept together with the
                        prediction scores.
  --predict_output_path 
                        If provided the prediction scores will be saved
                        to this path. Otherwise, files will be saved to
                        --output_path.
  --predict_output_format {parquet,csv,tsv}
                        Format of the output prediction files. By
                        default 'parquet', which is the most performant
                        format.
```                        