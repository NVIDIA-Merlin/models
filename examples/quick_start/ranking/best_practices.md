Describe best practices for:

# Preprocessing

## Data munging
- Converting data into the right shape: each each example is either a real (positive) or non-existing (negative) user-item interaction. You can see in the following example from TenRec dataset that your dataset might contain user and item features, and one or more targets, that can be either binary (for classification) or continuous/discrete (for regression).

![TenRec dataset structure](../images/tenrec_dataset.png)

- The input format can be CSV or Parquet, but the latter is recommended for being a columnar format which is faster to preprocess.

## Feature Engineering
- For count or long-tail distributions of continuous features, you might want to apply a log transformation before standardization. This can be done with NVTabular Log op.
- Count / Target encoding


## Filtering
- Filtering infrequent users (`--min_user_freq`) and items (`--min_item_freq`) is a common practice, as it is hard to learn good embeddings for them... Talk also about frequency capping/hashing alternatices...


## Data set splitting
- "random"
- "random_by_user"
- "temporal"


# Training

- stl_positive_class_weight
- Negative sampling: in_batch_negatives_train
- MTL aspects: --tasks_sample_space

Different ranking model characteristics
- STL: "mlp", "dcn", "dlrm", "deepfm", "wide_n_deep"
- MTL: "mmoe", "cgc","ple",


### Setting tasks sample space
In that dataset, some targets depend on others. For example, you only have a `like/follow/share=1` event if the user has clicked in the item. The learning of the dependent tasks is better if we set the appropriate sample space for the targets. In this case, we want to train the `click` target  using the entire space, and train the other targets (i.e., compute the loss) only for click space (where `click=1`).  

The scripts allows for setting the tasks sample space by using `--tasks_sample_space`, where the position should match the order of the `--tasks`. Empty value means the task will be trained in the entire space, i.e., loss computed for all examples in the dataset.

# Hyperparameter tuning

## W&B sweeps

