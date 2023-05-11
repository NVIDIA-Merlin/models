from rich.tree import Tree
from torch import nn


def module_tree(module: nn.Module, name: str = "") -> Tree:
    if hasattr(module, "__rich_repr__"):
        return module.__rich_repr__()

    tree = Tree(name or module._get_name())
    for child_name, child in module.named_children():
        if hasattr(child, "__rich_repr__"):
            tree.add(child.__rich_repr__())
        else:
            tree.add(module_tree(child, child_name))

    return tree
