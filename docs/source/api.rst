*****************
API Documentation
*****************


TensorFlow Models
------------------

.. currentmodule:: merlin.models.tf


Model Constructors
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Model
   DCNModel
   DLRMModel
   MatrixFactorizationModel
   TwoTowerModel
   YoutubeDNNRetrievalModel


Block Constructors
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   Block
   InputBlock
   CrossBlock
   DLRMBlock
   DotProductInteraction
   ParallelPredictionBlock
   MLPBlock
   DenseResidualBlock
   MMOEGate
   MMOEBlock
   MatrixFactorizationBlock
   DualEncoderBlock
   SequentialBlock
   ResidualBlock
   CGCBlock
   TabularBlock
   ContinuousFeatures
   EmbeddingFeatures
   SequenceEmbeddingFeatures
   TwoTowerBlock
   AsDenseFeatures
   AsSparseFeatures
   AsTabular
   Filter
   ItemRetrievalScorer
   StochasticSwapNoise
   FIFOQueue

Masking Block Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   CausalLanguageModeling
   MaskedLanguageModeling


Modeling Task Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: merlin.models.tf

.. autosummary::
   :toctree: generated

   PredictionTasks
   PredictionTask
   BinaryClassificationTask
   MultiClassClassificationTask
   RegressionTask
   ItemRetrievalTask
   NextItemPredictionTask


Transformation Block Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   ExpandDims


Functions for Blocks
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   ContinuousEmbedding


Dataset Constructors
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   merlin.models.tf.dataset.BatchedDataset

Metrics
~~~~~~~

.. autosummary::
   :toctree: generated

   NDCGAt
   AvgPrecisionAt
   RecallAt
   ranking_metrics

Sampling
~~~~~~~~

.. autosummary::
   :toctree: generated

   ItemSampler
   InBatchSampler
   CachedCrossBatchSampler
   CachedUniformSampler
   PopularityBasedSampler

 
Losses
~~~~~~

.. currentmodule:: merlin.models.tf.losses

.. autosummary::
   :toctree: generated

   CategoricalCrossEntropy
   SparseCategoricalCrossEntropy
   AdaptiveHingeLoss
   BPRLoss
   BPRmaxLoss
   HingeLoss
   LogisticLoss
   TOP1Loss
   TOP1maxLoss
   TOP1v2Loss
   loss_registry
 

Data
----

.. currentmodule:: merlin.models

.. autosummary::
   :toctree: generated

   merlin.models.data.synthetic.SyntheticData

Loaders
-------

.. autosummary::
   :toctree: generated

   merlin.models.loader.backend.ChunkQueue
   merlin.models.loader.backend.DataLoader
   merlin.models.loader.shuffle.Shuffle

Loader Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   merlin.models.loader.utils.device_mem_size

Loader Utility Functions for TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   merlin.models.loader.tf_utils.configure_tensorflow
   merlin.models.loader.tf_utils.get_dataset_schema_from_feature_columns


Utilities
---------


Miscellaneous Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   merlin.models.utils.misc_utils.filter_kwargs
   merlin.models.utils.misc_utils.safe_json
   merlin.models.utils.misc_utils.get_filenames
   merlin.models.utils.misc_utils.get_label_feature_name
   merlin.models.utils.misc_utils.get_timestamp_feature_name
   merlin.models.utils.misc_utils.get_parquet_files_names
   merlin.models.utils.misc_utils.Timing
   merlin.models.utils.misc_utils.get_object_size
   merlin.models.utils.misc_utils.validate_dataset


Registry Functions
------------------

.. autosummary::
   :toctree: generated

   merlin.models.utils.registry.camelcase_to_snakecase
   merlin.models.utils.registry.snakecase_to_camelcase
   merlin.models.utils.registry.default_name
   merlin.models.utils.registry.default_object_name
   merlin.models.utils.registry.Registry
   merlin.models.utils.registry.RegistryMixin
   merlin.models.utils.registry.display_list_by_prefix


Schema Functions
----------------

.. autosummary::
   :toctree: generated

   merlin.models.utils.schema_utils.select_targets
   merlin.models.utils.schema_utils.schema_to_tensorflow_metadata_json
   merlin.models.utils.schema_utils.tensorflow_metadata_json_to_schema
   merlin.models.utils.schema_utils.create_categorical_column
   merlin.models.utils.schema_utils.create_continuous_column
   merlin.models.utils.schema_utils.filter_dict_by_schema
   merlin.models.utils.schema_utils.categorical_cardinalities
   merlin.models.utils.schema_utils.categorical_domains
   merlin.models.utils.schema_utils.get_embedding_sizes_from_schema
   merlin.models.utils.schema_utils.get_embedding_size_from_cardinality
