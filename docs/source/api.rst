*****************
API Documentation
*****************


TensorFlow Models
------------------

.. currentmodule:: merlin.models.tf


Ranking Model Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   DCNModel
   DLRMModel

Retrieval Model Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   MatrixFactorizationModel
   TwoTowerModel
   YoutubeDNNRetrievalModel

Input Block Constructors
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated

   InputBlock
   ContinuousFeatures
   ContinuousEmbedding
   EmbeddingFeatures
   SequenceEmbeddingFeatures

Model Building Block Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   DLRMBlock
   MLPBlock
   CrossBlock
   TwoTowerBlock
   MatrixFactorizationBlock
   DotProductInteraction

Modeling Prediction Task Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   PredictionTasks
   PredictionTask
   BinaryClassificationTask
   MultiClassClassificationTask
   RegressionTask
   ItemRetrievalTask
   NextItemPredictionTask

Model Pipeline Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   SequentialBlock
   ParallelBlock
   ParallelPredictionBlock
   DenseResidualBlock
   DualEncoderBlock
   ResidualBlock
   TabularBlock
   Filter


Masking Block Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   CausalLanguageModeling
   MaskedLanguageModeling


Transformation Block Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   ExpandDims
   AsDenseFeatures
   AsSparseFeatures
   StochasticSwapNoise
   AsTabular

Multi-Task Block Constructors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated

   MMOEBlock
   CGCBlock

Data Loader Customization Constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
   BPRLoss
   BPRmaxLoss
   HingeLoss
   LogisticLoss
   TOP1Loss
   TOP1maxLoss
   TOP1v2Loss


Schema Functions
----------------

.. currentmodule:: merlin.models.utils

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

