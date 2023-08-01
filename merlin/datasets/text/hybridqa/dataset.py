import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from git import Repo
from tqdm import tqdm

import merlin.dtypes as md
import merlin.io
from merlin.core.dispatch import get_lib
from merlin.datasets import BASE_PATH
from merlin.models.tokenizers import Tokenizer
from merlin.models.tokenizers.sentencepiece import require_sentencepiece
from merlin.models.utils.doc_utils import docstring_parameter
from merlin.schema import ColumnSchema, Schema, Tags

DATASET_DIRNAME = "hybridqa"
HYBRID_QA_REPO = "https://github.com/wenhuchen/HybridQA.git"
HYBRID_QA_DIRNAME = "HybridQA"
WIKI_TABLES_REPO = "https://github.com/wenhuchen/WikiTables-WithLinks.git"
WIKI_TABLES_DIRNAME = "WikiTables-WithLinks"
DEFAULT_LLAMA_TOKENIZER_PATH = "llama/tokenizer.model"

_HYBRIDQA_REF = """
    [1] https://hybridqa.github.io/
"""
_WIKITABLES_REF = """
    [1] https://github.com/wenhuchen/WikiTables-WithLinks
"""


@docstring_parameter(hybridqa_ref=_HYBRIDQA_REF)
def get_hybridqa(
    tokenizer: Optional[Tokenizer] = None,
    path: Optional[Union[str, Path]] = None,
    tokenizer_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    **kwargs,
) -> Tuple[merlin.io.Dataset, merlin.io.Dataset, Dict[str, merlin.io.Dataset]]:
    """
    Downloads, preprocesses, and tokenizes the HybridQA dataset [1].

    Parameters
    ----------
    tokenizer : Tokenizer
        The tokenizer used to tokenize the questions and answers.
    path : Optional[Union[str, Path]], default=None
        The directory where the HybridQA dataset will be stored.
        If None, the dataset will be stored in the default directory.
    overwrite : bool, default=False
        If True, overwrite the existing downloaded files.

    Returns
    -------
    Tuple[merlin.io.Dataset, merlin.io.Dataset, merlin.io.Dataset]
        The preprocessed and tokenized datasets:
        table name to dataset mapping, training dataset, and test dataset.

    References
    ----------
    {hybridqa_ref}
    """
    if path is None:
        path = Path(BASE_PATH) / DATASET_DIRNAME
    else:
        path = Path(path)

    if tokenizer is None:
        tokenizer = load_llama_tokenizer(tokenizer_path or Path(DEFAULT_LLAMA_TOKENIZER_PATH))

    download_hybridqa(path, overwrite=overwrite)

    table_name_to_dataset = preprocess_hybridqa_tables(path, tokenizer)

    train, test = preprocess_hybridqa_questions(path, tokenizer)

    return train, test, table_name_to_dataset


def load_llama_tokenizer(
    tokenizer_path: Union[str, Path]
) -> "SentencePieceeTokenizer":  # noqa: F821
    """Loads the Llama tokenizer model file.

    Parameters
    ----------
    tokenizer_path : Path
        The path where the tokenizer model file is located.

    Returns
    -------
    SentencePieceTokenizer
    """
    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        raise RuntimeError(
            f"Failed to find tokenizer model at {tokenizer_path}. "
            "Define a custom `tokenizer`, or use the correct `tokenizer_path` "
            "that points to the Llama tokenizer."
        )

    require_sentencepiece()
    from sentencepiece import SentencePieceProcessor

    from merlin.models.tokenizers.sentencepiece import SentencePieceTokenizer

    processor = SentencePieceProcessor(model_file=str(tokenizer_path))
    tokenizer = SentencePieceTokenizer(processor=processor)

    return tokenizer


@docstring_parameter(
    hybridqa_ref=_HYBRIDQA_REF,
    wikitables_ref=_WIKITABLES_REF.replace("[1]", "[2]"),
)
def download_hybridqa(path: Path, overwrite: bool = False):
    """
    Automatically download the HybridQA [1] dataset and WikiTables [2] dataset to a given path.

    Parameters
    ----------
    path : Path
        The directory where the HybridQA and WikiTables datasets will be downloaded.
    overwrite : bool, default=False
        If True, overwrite the existing downloaded files.

    References
    ----------
    {hybridqa_ref}
    {wikitables_ref}
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    hybridqa_path = path / HYBRID_QA_DIRNAME
    if hybridqa_path.exists():
        return

    if overwrite:
        shutil.rmtree(hybridqa_path)

    Repo.clone_from(HYBRID_QA_REPO, hybridqa_path)

    wikitables_path = hybridqa_path / WIKI_TABLES_DIRNAME
    if wikitables_path.exists():
        return

    Repo.clone_from(WIKI_TABLES_REPO, wikitables_path)


def preprocess_hybridqa_questions(path: Path, tokenizer: Tokenizer) -> Dict[str, Any]:
    """
    Preprocess the HybridQA questions, tokenizing the question and answer text.

    Parameters
    ----------
    path : Path
        The directory where the HybridQA dataset is stored.
    tokenizer : Tokenizer
        The tokenizer used to tokenize the questions and answers.

    Returns
    -------
    Tuple[dict, dict]
        The tokenized training and test data.
    """

    train_json = path / HYBRID_QA_DIRNAME / "released_data" / "train.json"
    test_json = path / HYBRID_QA_DIRNAME / "released_data" / "test.json"

    with open(train_json) as f:
        train_raw = json.load(f)
    with open(test_json) as f:
        test_raw = json.load(f)

    train = _tokenize_question_and_answer(train_raw, tokenizer, train=True)
    test = _tokenize_question_and_answer(test_raw, tokenizer, train=False)

    return train, test


def preprocess_hybridqa_tables(
    path: Path, tokenizer: Tokenizer, overwrite: bool = False
) -> Dict[str, merlin.io.Dataset]:
    """
    Preprocess the HybridQA tables, encoding the table contents.

    Parameters
    ----------
    path : Path
        The directory where the HybridQA dataset is stored.
    tokenizer : Tokenizer
        The tokenizer used to encode the tables.
    overwrite : bool, default=False
        If True, overwrite the existing downloaded files.

    Returns
    -------
    dict
        A dictionary mapping table names to tokenized table data.
    """
    table_name_to_dataset = {}
    tables_path = path / HYBRID_QA_DIRNAME / WIKI_TABLES_DIRNAME / "tables_tok"
    request_path = path / HYBRID_QA_DIRNAME / WIKI_TABLES_DIRNAME / "request_tok"

    out_path = path / HYBRID_QA_DIRNAME / "tables"
    if overwrite:
        shutil.rmtree(out_path)
    out_path.mkdir(exist_ok=True)

    tables_files = sorted(f.name for f in os.scandir(tables_path))
    for table_file in tqdm(tables_files):
        table_id = table_file.rstrip(".json")
        parquet_path = out_path / f"{table_id}.parquet"
        if parquet_path.exists():
            dataset = merlin.io.Dataset(parquet_path, engine="parquet")
            table_name_to_dataset[table_id] = dataset
            continue

        table_file_path = tables_path / table_file
        with open(table_file_path) as f:
            table_json = json.load(f)
        request_file_path = request_path / table_file
        with open(request_file_path) as f:
            request_json = json.load(f)

        header = table_json["header"]
        column_names = []
        for col in header:
            name = col[0]
            # remove duplicate column names
            i = 0
            while name in column_names:
                name = f"{name}_{i}"
                i += 1
            column_names.append(name)

        raw_data = table_json["data"]
        transformed_data = defaultdict(list)
        for row in raw_data:
            for col_name, col_data in zip(column_names, row):
                data, links = col_data
                transformed_data[col_name].append(tokenizer.encode(data))
                if links:
                    transformed_data[col_name + " description"].append(
                        tokenizer.encode(" ".join(request_json[link] for link in links))
                    )
                else:
                    transformed_data[col_name + " description"].append(None)

        # drop empty columns
        transformed_data = {
            name: column
            for name, column in transformed_data.items()
            if not all(value is None for value in column)
        }

        df = get_lib().DataFrame(transformed_data)
        df.to_parquet(parquet_path)

        schema = Schema(
            [
                ColumnSchema(name=name, dtype=md.int32, tags=[Tags.TOKENIZED])
                for name in column_names
            ]
        )
        dataset = merlin.io.Dataset(parquet_path, schema=schema, engine="parquet")
        table_name_to_dataset[table_id] = dataset

    return table_name_to_dataset


def _tokenize_question_and_answer(raw_data, tokenizer, train=True):
    outputs = []
    for entry in raw_data:
        processed = {}
        processed["table_id"] = entry["table_id"]
        processed["question"] = tokenizer.encode(entry["question"])
        if train:
            processed["answer"] = tokenizer.encode(entry["answer-text"])
        outputs.append(processed)
    return outputs


def main():
    get_hybridqa(tokenizer_path="../llama/tokenizer.model", overwrite=True)


if __name__ == "__main__":
    main()
