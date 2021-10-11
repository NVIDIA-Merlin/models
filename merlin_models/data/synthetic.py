import logging

import numpy as np
import pandas as pd
from merlin_standard_lib import Schema, Tag
from merlin_standard_lib.utils.proto_utils import has_field

LOG = logging.getLogger("transformers4rec")


def generate_recsys_data(num_interactions: int, schema: Schema) -> pd.DataFrame:
    """
    Util function to generate synthetic data for session-based item-interactions
    from a schema object. It supports the generation of session and item features.
    The schema should include a few tags:

    - `Tag.SESSION` for features related to sessions
    - `Tag.SESSION_ID` to tag the session-id feature
    - `Tag.ITEM` for features related to item interactions.

    Parameters:
    ----------
    num_interactions: int
        number of interaction rows to generate.
    schema: Schema
        schema object describing the columns to generate.

    Returns
    -------
    data: pd.DataFrame
        Pandas dataframe with synthetic generated data.
    """
    data = pd.DataFrame(
        [i for i in range(num_interactions)],
        columns=["session-id"],
    ).astype(np.int64)

    item_id_col = schema.select_by_tag(Tag.ITEM_ID).feature[0]
    data[item_id_col.name] = np.clip(
        np.random.lognormal(3.0, 1.0, num_interactions).astype(np.int32),
        1,
        item_id_col.int_domain.max,
    ).astype(np.int64)

    # get item-id cols
    features = schema.remove_by_tag(Tag.ITEM_ID).feature
    for feature in features:
        is_int_feature = has_field(feature, "int_domain")
        if is_int_feature:
            data[feature.name] = pd.cut(
                data[item_id_col.name],
                bins=feature.int_domain.max - 1,
                labels=np.arange(1, feature.int_domain.max),
            ).astype(np.int64)
        else:
            data[feature.name] = np.random.uniform(
                feature.float_domain.min, feature.float_domain.max, num_interactions
            )

    return data
