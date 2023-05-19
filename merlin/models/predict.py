from typing import Callable, Generic, TypeVar

import numpy as np

import merlin
from merlin.core.dispatch import DataFrameType, concat_columns, get_lib
from merlin.schema import Schema

PREDICTION_SUFFIX = "_"


def safe_concat_columns(input_df: DataFrameType, output_df: DataFrameType) -> DataFrameType:
    """Safely concatenate columns from two dataframes. 
    
    If the column names overlap, the ones in the output_df are renamed
    by appending a suffix.

    Parameters
    ----------
    input_df : DataFrameType
        Input dataframe.
    output_df : DataFrameType
        Output dataframe.

    Returns
    -------
    DataFrameType
        Concatenated dataframe.
    """
    
    input_col_set = set(input_df.columns)

    _to_rename = [col for col in output_df.columns if col in input_col_set]
    if _to_rename:
        output_df = output_df.rename(
            columns={col: f"{col}{PREDICTION_SUFFIX}" for col in _to_rename}
        )

    return concat_columns([input_df, output_df])


ModelT = TypeVar("ModelT")


class ModelEncode(Generic[ModelT]):
    """
    A class for model encoding. This class provides methods to load a model, process data, and encode the output
    into a desired schema.

    Parameters
    ----------
    model : ModelT
        The model to be used for encoding.
    output_schema : Schema
        The desired output schema.
    data_iterator_func : callable, optional
        Function for data iteration.
    model_load_func : callable, optional
        Function for model loading.
    model_encode_func : callable, optional
        Function for model encoding.
    output_concat_func : callable, optional
        Function for output concatenation.
    post : Callable[[DataFrameType, DataFrameType], DataFrameType], optional
        Post-processing function.
    """
    
    def __init__(
        self,
        model: ModelT,
        output_schema: Schema,
        data_iterator_func=None,
        model_load_func=None,
        model_encode_func=None,
        output_concat_func=None,
        post: Callable[[DataFrameType, DataFrameType], DataFrameType] = safe_concat_columns,
    ):
        super().__init__()
        self._model = model
        self.output_schema = output_schema
        self.data_iterator_func = data_iterator_func
        self.model_load_func = model_load_func
        self.model_encode_func = model_encode_func
        self.post = post

        if not output_concat_func:
            if output_schema and len(output_schema) == 1:  # type: ignore
                output_concat_func = np.concatenate
            else:
                output_concat_func = get_lib().concat  # type: ignore

        self.output_concat_func = output_concat_func

    @property
    def model(self) -> ModelT:
        if isinstance(self._model, str):
            self._model = self.model_load_func(self._model)
        return self._model

    def __call__(
        self,
        df: DataFrameType,
        add_inputs: bool = True,
        index=None,
    ) -> DataFrameType:
        """Apply the model encoding on the input dataframe. 
        
        This method processes the dataframe in batches, encodes 
        each batch using the model, and then concatenates the results. 
        If `add_inputs` is True, it also concatenates the input dataframe 
        with the encoded output. 
        If `index` is provided, the corresponding column of the input 
        dataframe is added to the output.

        Parameters
        ----------
        df : DataFrameType
            The input dataframe to be encoded.
        add_inputs : bool, default True
            Whether to concatenate the input dataframe with the model's output.
        index : str or None, default None
            The name of the column in the input dataframe to be added to the output dataframe.

        Returns
        -------
        DataFrameType
            The output dataframe after model encoding.

        Raises
        ------
        ValueError
            If the output schema is neither 1D nor 2D.
        """
        
        # Set defaults
        iterator_func = self.data_iterator_func or (lambda x: [x])
        encode_func = self.model_encode_func or (lambda x, y: x(y))
        concat_func = self.output_concat_func or np.concatenate

        # Iterate over batches of df and collect predictions
        outputs = concat_func([encode_func(self.model, batch) for batch in iterator_func(df)])

        if len(self.output_schema) == 1:
            shape = self.output_schema.first.shape

            if len(shape.dims) == 1:
                output_names = self.output_schema.column_names
            if len(shape.dims) == 2:
                output_names = [str(i) for i in range(outputs.shape[1])]
            else:
                raise ValueError("Only 1D or 2D outputs are supported")
        else:
            output_names = self.output_schema.column_names
        model_output_df = type(df)(outputs, columns=output_names)

        if add_inputs:
            output_df = self.post(df, model_output_df)
        else:
            output_df = model_output_df
            if index:
                output_df[index] = df[index]

        return output_df

    def transform(self, col_selector, df: DataFrameType, **kwargs) -> DataFrameType:
        return self(df[col_selector], **kwargs)

    def encode_dataset(
        self, dataset: merlin.io.Dataset, add_inputs: bool = True, index=None
    ) -> merlin.io.Dataset:
        """Apply the model encoding on a merlin.io.Dataset or dask DataFrame. 
        
        This method maps the __call__ method to each partition of the dataset. 
        If `add_inputs` is True, it also concatenates the input dataset with 
        the encoded output. 
        If `index` is provided, the corresponding column of the input dataset is added to 
        the output.

        Parameters
        ----------
        dataset : merlin.io.Dataset
            The input dataset to be encoded.
        add_inputs : bool, default True
            Whether to concatenate the input dataset with the model's output.
        index : str or None, default None
            The name of the column in the input dataset to be added to the output dataframe.

        Returns
        -------
        merlin.io.Dataset
            The output dataset after model encoding.

        Raises
        ------
        ValueError
            If the input is neither a merlin.io.Dataset nor a dask DataFrame.
        """
        
        # Check if merlin-dataset is passed
        if hasattr(dataset, "to_ddf"):
            dataset = dataset.to_ddf()

        if not hasattr(dataset, "map_partitions"):
            raise ValueError("Please pass in a merlin.io.Dataset or a dask-DataFrame")

        predictions = dataset.map_partitions(self, index=index, add_inputs=add_inputs)

        if index is not None:
            predictions = predictions.set_index(index)

        return merlin.io.Dataset(predictions)

    @staticmethod
    def create_data_iterator_func(loader_cls, batch_size: int):
        """Create a data iterator function for a given dataloader class and batch size.

        This method constructs a function that, when called with a dataset, will produce an iterator that yields 
        batches of the specified size using the specified loader class. This is useful for processing large datasets 
        in chunks, especially in the context of machine learning where it's common to work with data in batches.

        Parameters
        ----------
        loader_cls : class
            The class to be used for loading data. This class should accept a merlin.io.Dataset and a batch size 
            as its arguments.
        batch_size : int
            The size of the batches to be yielded by the data iterator.

        Returns
        -------
        function
            The data iterator function.
        """
        
        def data_iterator(dataset):
            return loader_cls(
                merlin.io.dataset.Dataset(dataset),
                batch_size=batch_size,
            )

        return data_iterator
