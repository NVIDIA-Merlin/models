from mypy.options import Options
from mypy.plugin import FunctionContext, Plugin
from mypy.types import Type


def plugin(version: str):
    return MerlinModelsPlugin


class MerlinModelsPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        super().__init__(options)

    def get_method_hook(self, fullname: str):
        if fullname.startswith("merlin.models.tf") and fullname.endswith("connect"):
            return merlin_connect_hook_tf
        if fullname.startswith("merlin.models.tf") and fullname.endswith("connect_branch"):
            return merlin_connect_hook_tf

        return default_function_hook


def default_function_hook(ctx: FunctionContext) -> Type:
    return ctx.default_return_type


def merlin_connect_hook_tf(ctx: FunctionContext) -> Type:
    SequentialBlock, Model, RetrievalModel = ctx.default_return_type.items  # type: ignore

    model_like_blocks = [
        "merlin.models.tf.core.ParallelPredictionBlock",
        "merlin.models.tf.core.PredictionTask",
        "merlin.models.tf.prediction.classification.BinaryClassificationTask",
        "merlin.models.tf.prediction.regression.RegressionTask",
    ]

    retrieval_blocks = [
        "merlin.models.tf.block.retrieval.TwoTowerBlock",
        "merlin.models.tf.block.retrieval.MatrixFactorizationBlock",
    ]

    if str(ctx.arg_types[0][-1]) in model_like_blocks:
        if str(ctx.type) in retrieval_blocks:  # type: ignore
            return RetrievalModel

        return Model

    return SequentialBlock
