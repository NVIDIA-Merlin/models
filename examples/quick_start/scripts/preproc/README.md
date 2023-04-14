# Preprocessing script
The `preprocessing.py` is a template script that provides basic preprocessing and feature engineering operations for tabular data. It uses the [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) and [dask-cudf](https://github.com/rapidsai/cudf/tree/main/python/dask_cudf) libraries for GPU accelerated preprocessing.

In this document we describe the provided preprocessing and feature engineernig options and the corresponding command line arguments.

## Best practices
TODO: List best practices on preprocessing and feature engineering

### Data munging
- Converting data into the right shape: each each example is either a real (positive) or non-existing (negative) user-item interaction. You can see in the following example from TenRec dataset that your dataset might contain user and item features, and one or more targets, that can be either binary (for classification) or continuous/discrete (for regression).

![TenRec dataset structure](../../images/tenrec_dataset.png)

- The input format can be CSV or Parquet, but the latter is recommended for being a columnar format which is faster to preprocess.

### Feature Engineering
- For count or long-tail distributions of continuous features, you might want to apply a log transformation before standardization. This can be done with NVTabular Log op.
- Count / Target encoding


### Filtering
- Filtering infrequent users (`--min_user_freq`) and items (`--min_item_freq`) is a common practice, as it is hard to learn good embeddings for them... Talk also about frequency capping/hashing alternatices...


### Data set splitting
- "random"
- "random_by_user"
- "temporal"

## Command line arguments
### Inputs
```
  --data_path
                        Path to the data
  --eval_data_path 
                        Path to eval data, if data was already splitMust have
                        the same schema as train data (in --data_path).
  --test_data_path 
                        Path to test data, if data was already split. This
                        data is expected to have the same input features as
                        train data but targets are not necessary, as this data
                        is typically used for prediction.
  --input_data_format {csv,tsv,parquet}
                        Input data format
  --csv_sep             Character separator for CSV files.Default is ','. You
                        can use 'tab' for tabular separated data, or
                        --input_data_format tsv
  --csv_na_values 
                        String in the original data that should be replaced by
                        NULL
```

### Outputs
```
  --output_path 
                        Output path where the preprocessed files will be
                        savedDefault is ./results/
  --output_num_partitions 
                        Number of partitions that result in this number of
                        output filesDefault is 10.
  --persist_intermediate_files 
                        Whether to persist/cache the intermediate
                        preprocessing files. Enabling this might be necessary
                        for larger datasets.
```

### Features and targets definition
```
  --control_features 
                        Columns (comma-separated) that should be kept as is in
                        the output files. For example,
                        --control_features=session_id,timestamp
  --categorical_features 
                        Columns (comma-sep) with categorical/discrete features
                        that will encoded/categorified to contiguous ids in
                        the preprocessing. These tags are tagged as
                        'categorical' in the schema, so that Merlin Models can
                        automatically create embedding tables for them.
  --continuous_features 
                        Columns (comma-sep) with continuous features that will
                        be standardized and tagged in the schema as
                        'continuous', so that the Merlin Models can represent
                        and combine them with embedding properly.
  --continuous_features_fillna 
                        Replaces NULL values with this float. You can also set
                        it with 'median' for filling nulls with the median
                        value.
  --user_features 
                        Columns (comma-sep) that should be tagged in the
                        schema as user features. This information might be
                        useful for modeling later.
  --item_features 
                        Columns (comma-sep) that should be tagged in the
                        schema as item features. This information might be
                        useful for modeling later, for example, for in-batch
                        sampling if your data contains only positive examples.
  --user_id_feature 
                        Column that contains the user id feature, for tagging
                        in the schema. This information is used in the
                        preprocessing if you set --min_user_freq or
                        --max_user_freq
  --item_id_feature 
                        Column that contains the item id feature, for tagging
                        in the schema. This information is used in the
                        preprocessing if you set --min_item_freq or
                        --max_item_freq
  --timestamp_feature 
                        Column containing a timestamp or date feature. The
                        basic preprocessing doesn't extracts date and time
                        features for it. It is just tagged as 'timestamp' in
                        the schema and used for splitting train / eval data if
                        --dataset_split_strategy=temporal is used.
  --session_id_feature SESSION_ID_FEATURE
                        This is just for tagging this feature.
  --binary_classif_targets 
                        Columns (comma-sep) that should be tagged in the
                        schema as binary target. Merlin Models will create a
                        binary classification head for each of these targets.
  --regression_targets 
                        Columns (comma-sep) that should be tagged in the
                        schema as binary target. Merlin Models will create a
                        regression head for each of these targets.
```

### Data casting and filtering
```
  --to_int32            Cast these columns (comma-sep) to int32.
  --to_int16            Cast these columns (comma-sep) to int16, to save some
                        memory.
  --to_int8             Cast these columns (comma-sep) to int32, to save some
                        memory.
  --to_float32 
                        Cast these columns (comma-sep) to float32
  --min_user_freq 
                        Users with frequency lower than this value are removed
                        from the dataset (before data splitting).
  --max_user_freq 
                        Users with frequency higher than this value are
                        removed from the dataset (before data splitting).
  --min_item_freq 
                        Items with frequency lower than this value are removed
                        from the dataset (before data splitting).
  --max_item_freq 
                        Items with frequency higher than this value are
                        removed from the dataset (before data splitting).
  --num_max_rounds_filtering 
                        Max number of rounds interleaving users and items
                        frequency filtering. If a small number of rounds is
                        chosen, some low-frequent users or items might be kept
                        in the dataset. Default is 5
  --filter_query 
                        A filter query condition compatible with dask-cudf
                        `DataFrame.query()`
```

### Dataset splitting (train and eval sets)
```
  --dataset_split_strategy {random,random_by_user,temporal}
                        If None, no data split is performed. If 'random',
                        samples are assigned randomly to eval set according to
                        --random_split_eval_perc. If 'random_by_user', users
                        will have examples in both train and eval set,
                        according to the proportion specified in
                        --random_split_eval_perc. If 'temporal', the
                        --timestamp_feature with
                        --dataset_split_temporal_timestamp to split eval set
                        based on time.
  --random_split_eval_perc 
                        Percentage of examples to be assigned to eval set. It
                        is used with --dataset_split_strategy 'random' and
                        'random_by_user'
  --dataset_split_temporal_timestamp 
                        Used when --dataset_split_strategy 'temporal'. It
                        assigns for eval set all examples where the
                        --timestamp_feature >= value
```

### CUDA cluster options
```                      
  --visible_gpu_devices 
                        Ids of GPU devices that should be used for
                        preprocessing, if any. For example:
                        --visible_gpu_devices=0,1. Default is 0
  --gpu_device_spill_frac 
                        Percentage of GPU memory used at which
                        LocalCUDACluster should spill memory to CPU, before
                        raising out-of-memory errors. Default is 0.7
```