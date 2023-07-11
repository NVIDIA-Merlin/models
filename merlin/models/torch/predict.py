from functools import partial, reduce
from typing import Dict, Optional, TypeVar, Union, overload

import torch
from torch import nn

from merlin.core.dispatch import DataFrameLike, concat, concat_columns
from merlin.dataloader.torch import Loader
from merlin.io import Dataset
from merlin.models.torch.batch import Batch
from merlin.models.torch.block import BatchBlock, Block
from merlin.models.torch.schema import Selection, select
from merlin.schema import ColumnSchema, Schema
from merlin.table import TensorTable

OUT_KEY = "output"
DFType = TypeVar("DFType", bound=DataFrameLike)


class EncoderBlock(Block):
    """
    A block that runs a `BatchBlock` as a pre-processing step before running the rest.

    This ensures that the batch is created at inference time as well.

    Parameters
    ----------
    *module : nn.Module
        Variable length argument list of PyTorch modules.
    pre : BatchBlock, optional
        An instance of BatchBlock class for pre-processing.
        If None, an instance of BatchBlock is created.
    name : str, optional
        A name for the encoder block.

    Raises
    ------
    ValueError
        If the 'pre' argument is not an instance of BatchBlock.
    """

    def __init__(
        self, *module: nn.Module, pre: Optional[BatchBlock] = None, name: Optional[str] = None
    ):
        super().__init__(*module, name=name)
        if isinstance(pre, BatchBlock):
            self.pre = pre
        elif pre is None:
            self.pre = BatchBlock()
        else:
            raise ValueError(f"Invalid pre: {pre}, must be a BatchBlock")

    def forward(
        self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], batch: Optional[Batch] = None
    ):
        _batch: Batch = self.pre(inputs, batch=batch)

        outputs = _batch.inputs()
        for block in self.values:
            outputs = block(outputs, batch=_batch)
        return outputs

    @torch.jit.unused
    def encode(
        self,
        data: Union[Dataset, Loader, Batch],
        selection: Optional[Selection] = None,
        batch_size=None,
        index: Optional[Selection] = None,
        unique: bool = True,
    ):
        """
        Encodes a given data set.

        Parameters
        ----------
        data : Union[Dataset, Loader, Batch]
            Input data to encode.
        selection : Selection, optional
            Features to encode. If not provided, all features will be encoded.
        batch_size : int, optional
            Size of the batch to encode.
        index : Selection, optional
            Index selection.
        unique : bool, optional
            If True, only unique entries are returned.

        Returns
        -------
        encoded_data : Dataset
            Encoded data set.
        """
        _dask_encoder = DaskEncoder(self, selection=selection)

        return _dask_encoder(data, batch_size=batch_size, index=index, unique=unique)

    @torch.jit.unused
    def predict(
        self,
        data: Union[Dataset, Loader, Batch],
        selection: Optional[Selection] = None,
        batch_size=None,
        index: Optional[Selection] = None,
        prediction_suffix: str = "_prediction",
        unique: bool = True,
    ):
        """
        Encodes a given data set and predicts the output.
        All input-features will be present in the output.

        Parameters
        ----------
        data : Union[Dataset, Loader, Batch]
            Input data to encode.
        selection : Selection, optional
            Features to encode. If not provided, all features will be encoded.
        batch_size : int, optional
            Size of the batch to encode.
        index : Selection, optional
            Index selection.
        prediction_suffix : str, optional
            The suffix to add to the prediction columns in the output DataFrame.
        unique : bool, optional
            If True, only unique entries are returned.

        Returns
        -------
        predictions : dask_cudf.DataFrame
            Predictions of the data set.
        """
        _dask_predictor = DaskPredictor(
            self, prediction_suffix=prediction_suffix, selection=selection
        )

        return _dask_predictor(data, batch_size=batch_size, index=index, unique=unique)


class DaskEncoder:
    """Encode various forms of data using a specified PyTorch module.

    Supporting multiple data formats like Datasets, Loaders, DataFrames,
    and PyTorch tensors.

    Example usage for encoding with an index & selection::
        >>> dataset = Dataset(...)
        >>> model = mm.TwoTowerModel(dataset.schema)

        # `selection=Tags.USER` ensures that only the sub-module(s) of the model
        # that processes features tagged as user is used during encoding.
        # Additionally, it filters out all other features that aren't tagged as user.
        >>> user_encoder = DaskEncoder(model[0], selection=Tags.USER)

        # The index is used in the resulting DataFrame after encoding
        # Setting unique=True (default value) ensures that any duplicate rows
        # in the DataFrame, based on the index, are dropped, leaving only the
        # first occurrence.
        >>> user_embs = user_encoder(dataset, batch_size=128, index=Tags.USER_ID)
        >>> print(user_embs.compute())
        user_id    0         1         2    ...   37        38        39        40
        0       ...  0.1231     0.4132    0.5123  ...  0.9132    0.8123    0.1123
        1       ...  0.1521     0.5123    0.6312  ...  0.7321    0.6123    0.2213
        ...     ...  ...        ...       ...     ...  ...       ...       ...

    Parameters
    ----------
    module : nn.Module
        The PyTorch module used for encoding.
    selection : Optional[Selection], optional
        The data selection used for encoding, by default None.
    """

    def __init__(self, module: nn.Module, selection: Optional[Selection] = None):
        self.module = module
        self.selection = selection

    @overload  # pragma: no cover
    def __call__(
        self, data: Dataset, batch_size=None, index: Optional[Selection] = None, unique: bool = True
    ):
        ...

    @overload  # pragma: no cover
    def __call__(self, data: Loader, index: Optional[Selection] = None, unique: bool = True):
        ...

    @overload  # pragma: no cover
    def __call__(self, data: DataFrameLike, batch_size=None):
        ...

    @overload  # pragma: no cover
    def __call__(self, data: torch.Tensor):
        ...

    @overload  # pragma: no cover
    def __call__(self, data: Dict[str, torch.Tensor]):
        ...

    def __call__(self, data, batch_size=None, index=None, unique=True):
        """Encode a Dataset, Loader, DataFrame, or Tensor(s).

        Parameters
        ----------
        data : Dataset, Loader, DataFrameLike, torch.Tensor or Dict[str, torch.Tensor]
            The data to be encoded.
        batch_size : int, optional
            The batch size for the encoding, by default None.
        index : Optional[Selection], optional
            The data selection used for the encoding, by default None.
        unique : bool, optional
            If True, duplicate rows in the DataFrame are removed, by default True.
        """
        if isinstance(data, (Dataset, Loader)):
            return self.encode_dataset(data, batch_size, index=index, unique=unique)
        if isinstance(data, DataFrameLike):
            return self.encode_df(data, batch_size)
        if isinstance(data, (dict, torch.Tensor)):
            return self.encode_tensors(data)

        raise ValueError("data must be a DataFrameLike, a Dataset, or a Loader")

    def encode_dataset(
        self,
        data: Union[Dataset, Loader],
        batch_size: Optional[int] = None,
        index: Optional[Selection] = None,
        unique: bool = True,
    ) -> Dataset:
        """Encode a Dataset or Loader through Dask.

        Encoding happens in 3 steps:
        1. Partition Mapping
            This step uses Dask to break down the DataFrame into several partitions,
            making large datasets computationally manageable.
            The `call_df` function is applied to each partition independently,
            facilitating efficient distributed computation.
        2. DataFrame Processing
            In this step, each partition, which is a DataFrame, is transformed directly
            into a Loader with a determined batch size. This Loader then efficiently
            converts the data into batches of PyTorch tensors, which are subsequently
            processed by the PyTorch module using the `call_tensors` function.
        3. Tensor Processing
            Here, each batch derived from the Loader is processed by a PyTorch module for encoding.
            If the inputs are dictionary-like and a passthrough_schema is provided, supplementary
            columns might be included in the output DataFrame.

        Parameters
        ----------
        data : Union[Dataset, Loader]
            The data to be encoded.
        batch_size : Optional[int], optional
            The batch size for the encoding, by default None.
        index : Optional[Selection], optional
            The data selection used for the encoding, by default None.
        unique : bool, optional
            If True, duplicate rows in the DataFrame are removed, by default True.
        """
        if isinstance(data, Loader):
            batch_size = data.batch_size
            schema = data.input_schema
            dataset: Dataset = data.dataset
        elif isinstance(data, Dataset):
            if not batch_size:
                raise ValueError("batch_size must be provided if a Dataset is passed")
            schema = data.schema
            dataset: Dataset = data
        else:
            raise ValueError("data must be a DataFrameLike, a Dataset, or a Loader")

        if self.selection:
            schema = select(schema, self.selection)
            dataset = Dataset(dataset.to_ddf(), schema=schema)
            ddf = dataset.to_ddf()[schema.column_names]
        else:
            ddf = dataset.to_ddf()

        index_schema = None
        if index:
            index_schema = select(schema, index)

            if unique:
                ddf = ddf.drop_duplicates(index_schema.column_names, keep="first")

        output_schema = self.encoded_schema(
            ddf.head(), input_schema=schema, passthrough_schema=index_schema
        )
        output_dtypes = {col.name: col.dtype.to("numpy") for col in output_schema}

        output = ddf.map_partitions(
            self.encode_df,
            batch_size=batch_size,
            input_schema=schema,
            passthrough_schema=index_schema,
            meta=output_dtypes,
        )

        if index:
            output = output.set_index(index_schema.column_names)
            output_schema = output_schema.excluding_by_name(index_schema.column_names)

        return Dataset(output, schema=output_schema)

    def _module_schema(self, sample_df: DFType) -> Schema:
        """Get encoder module output schema.

        Parameters
        ----------
        sample_df : DFType
            Sample DataFrame containing inputs to the module.

        Returns
        -------
        Schema
            Output Schema describing module output
        """
        module_schema = None
        sample_x, _ = Loader(Dataset(sample_df), 2).peek()
        sample_output = self.module(sample_x)
        if hasattr(self.module, "output_schema"):
            module_schema = self.module.output_schema()
        else:
            sample_output_table = to_tensor_table(sample_output)
            module_schema = Schema(
                [
                    ColumnSchema(column, dtype=sample_output_table[column].dtype)
                    for column in sample_output_table.columns
                ]
            )
        return module_schema

    def encoded_schema(
        self, sample_df: DFType, input_schema: Schema, passthrough_schema: Schema
    ) -> Schema:
        """Return the output schema corresponding to the output

        Parameters
        ----------
        sample_df : DFType
            dataframe with sample of inputs to encoder module
        input_schema : Schema
            schema describing inputs
        passthrough_schema : Schema
            schema describing inputs passed through to the output dataframe

        Returns
        -------
        Schema
            schema describing output dataframe
        """
        module_schema = self._module_schema(sample_df)

        if passthrough_schema:
            output_schema = passthrough_schema + module_schema
        else:
            output_schema = module_schema

        # Ensure order of schema columns match output dataframe
        sample_output_df = self.encode_df(
            sample_df,
            batch_size=2,
            input_schema=input_schema,
            passthrough_schema=passthrough_schema,
        )
        output_schema = Schema([output_schema[column] for column in sample_output_df.columns])

        return output_schema

    def encode_df(
        self,
        df: DFType,
        batch_size: Optional[int] = None,
        input_schema: Optional[Schema] = None,
        passthrough_schema: Optional[Schema] = None,
    ) -> DFType:
        """Encode a DataFrame, either from Pandas or CuDF.

        Parameters
        ----------
        df : DFType
            The DataFrame to be encoded.
        batch_size : Optional[int], optional
            The batch size for the encoding, by default None.
        input_schema : Optional[Schema], optional
            The schema of the input DataFrame, by default None.
        passthrough_schema : Optional[Schema], optional
            The schema that should pass through the encoding, by default None.

        Returns
        -------
        DFType
            Encoded output DataFrame
        """
        dataset = Dataset(df, schema=input_schema)
        loader = Loader(dataset, batch_size=batch_size or len(df))
        apply = partial(self.encode_tensors, passthrough_schema=passthrough_schema)
        output_df = reduce(self.reduce, loader.map(apply))

        return output_df

    def encode_tensors(
        self, inputs, targets=None, passthrough_schema: Optional[Schema] = None
    ) -> DFType:
        """Encode a batch of Pytorch tensor(s).

        Parameters
        ----------
        inputs
            The inputs to be encoded.
        targets, optional
            The targets to be encoded, by default None.
        passthrough_schema : Optional[Schema], optional
            The schema that should pass through the encoding, by default None.
        """
        del targets
        output_df = to_tensor_table(self.module(inputs)).to_df()

        if passthrough_schema and isinstance(inputs, dict):
            col_names = passthrough_schema.column_names
            index_dict = {n: inputs[n] for n in col_names}
            index_df = to_tensor_table(index_dict).to_df()

            output_df = concat_columns([index_df, output_df])

        return output_df

    def reduce(self, left: DFType, right: DFType):
        """
        Concatenate two DataFrames along the index axis.

        Parameters
        ----------
        left : DFType
            The first DataFrame.
        right : DFType
            The second DataFrame.
        """
        return concat([left, right])


class DaskPredictor(DaskEncoder):
    """Prediction on various forms of data using a specified PyTorch module.

    This is especially useful when you want to keep track of both the original data and
    the predictions in one place, or when you need to perform further computations using
    both inputs and predictions.

    Example usage::
        >>> dataset = Dataset(...)
        >>> model = mm.TwoTowerModel(dataset.schema)
        >>> predictor = DaskPredictor(model)
        >>> predictions = predictor(dataset, batch_size=128)
        >>> print(predictions.compute())
        user_id  user_age  item_id  item_category  click  click_prediction
        0        24        101      1             1      0.6312
        1        35        102      2             0      0.7321
        ...      ...       ...      ...           ...    ...


    Parameters
    ----------
    module : nn.Module
        The PyTorch module used to transform the input tensors.
    selection : Selection, optional
        Selection of features to encode, if not provided, all features will be encoded.
    prediction_suffix : str, optional
        The suffix to add to the prediction columns in the output DataFrame.
    """

    def __init__(
        self,
        module: nn.Module,
        selection: Optional[Selection] = None,
        prediction_suffix: str = "_prediction",
    ):
        super().__init__(module, selection)
        self.prediction_suffix = prediction_suffix

    def encode_tensors(self, inputs, targets=None, **kwargs) -> DFType:
        """Encode a batch of Pytorch tensor(s), outputs include both inputs and predictions.

        Parameters
        ----------
        inputs :
            Input tensors to be transformed.
        targets : optional
            Target tensors.

        Returns
        -------
        output_df : DFType
            The output DataFrame.
        """

        del kwargs  # Unused since we pass-through everything

        input_df = to_tensor_table(inputs, "input").to_df()
        if targets is not None:
            target_df = to_tensor_table(targets, "target").to_df()
            output_df = safe_concat_columns(input_df, target_df, rename_suffix="_target")
        else:
            output_df = input_df

        module_df = to_tensor_table(self.module(inputs)).to_df()
        output_df = safe_concat_columns(output_df, module_df, self.prediction_suffix)

        return output_df

    def encoded_schema(
        self, sample_df: DFType, input_schema: Schema, passthrough_schema: Schema
    ) -> Schema:
        module_schema = self._module_schema(sample_df)

        renamed_module_schema = Schema()
        for col in module_schema:
            name = col.name
            if col.name in input_schema.column_names:
                name += self.prediction_suffix
            renamed_module_schema[name] = col.with_name(name)

        output_schema = input_schema + renamed_module_schema

        sample_output_df = self.encode_df(
            sample_df,
            batch_size=2,
            input_schema=input_schema,
            passthrough_schema=passthrough_schema,
        )
        output_schema = Schema([output_schema[column] for column in sample_output_df.columns])

        return output_schema


def to_tensor_table(
    data: Union[torch.Tensor, Dict[str, torch.Tensor]], default_key: str = OUT_KEY
) -> TensorTable:
    if isinstance(data, dict):
        return TensorTable(data)
    else:
        return TensorTable({default_key: data})


def safe_concat_columns(left: DFType, right: DFType, rename_suffix: str = "_") -> DFType:
    """Safely concatenate columns from two dataframes.

    If the column names overlap, the ones in the output_df are renamed
    by appending a suffix.

    Parameters
    ----------
    input_df : DataFrameType
        Input dataframe.
    output_df : DataFrameType
        Output dataframe.
    rename_suffix : str, optional
        Suffix to append to the column names in the output_df, by default "_"

    Returns
    -------
    DataFrameType
        Concatenated dataframe.
    """

    left_col_set = set(left.columns)

    _to_rename = [col for col in right.columns if col in left_col_set]
    if _to_rename:
        right = right.rename(columns={col: f"{col}{rename_suffix}" for col in _to_rename})

    return concat_columns([left, right])
