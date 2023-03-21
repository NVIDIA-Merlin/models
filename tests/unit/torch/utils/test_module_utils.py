import torch
import torch.nn as nn

from merlin.models.torch.utils.module_utils import (
    apply,
    find_all_instances,
    get_all_children,
    has_custom_call,
)


class NoArgsModule(nn.Module):
    def forward(self, x):
        return x * 2


class ArgsModule(nn.Module):
    def forward(self, x, factor):
        return x * factor


class KwargsModule(nn.Module):
    def forward(self, x, factor=1, add=0):
        return x * factor + add


class CustomCallModule(nn.Module):
    def forward(self, x):
        return x * 2

    def __call__(self, x):
        return self.forward(x) + 1


class Test_apply:
    def test_no_args_module(self):
        module = NoArgsModule()
        x = torch.tensor([1, 2, 3])
        y = apply(module, x)

        assert torch.allclose(y, x * 2)

    def test_args_module(self):
        module = ArgsModule()
        x = torch.tensor([1, 2, 3])
        y = apply(module, x, 3)

        assert torch.allclose(y, x * 3)

    def test_kwargs_module(self):
        module = KwargsModule()
        x = torch.tensor([1, 2, 3])
        y = apply(module, x, factor=3, add=5)

        assert torch.allclose(y, x * 3 + 5)

    def test_custom_call_module(self):
        module = CustomCallModule()
        x = torch.tensor([1, 2, 3])
        y = apply(module, x)

        assert torch.allclose(y, x * 2 + 1)
        assert has_custom_call(module)

    def test_has_custom_call(self):
        no_args_module = NoArgsModule()
        args_module = ArgsModule()
        kwargs_module = KwargsModule()
        custom_call_module = CustomCallModule()

        assert not has_custom_call(no_args_module)
        assert not has_custom_call(args_module)
        assert not has_custom_call(kwargs_module)
        assert has_custom_call(custom_call_module)


class Test_get_all_children:
    def test_sequential(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Sequential(
                nn.Linear(20, 30),
                nn.ReLU(),
                nn.Linear(30, 40),
            ),
        )

        all_children_modules = get_all_children(model)
        assert len(all_children_modules) == 6

    def test_custom_model(self):
        class CustomModel(nn.Module):
            def __init__(self):
                super(CustomModel, self).__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.ReLU()
                self.submodule = nn.Sequential(
                    nn.Linear(20, 30),
                    nn.ReLU(),
                    nn.Linear(30, 40),
                )

            def forward(self, x):
                pass

        custom_model = CustomModel()
        all_children_modules = get_all_children(custom_model)
        assert len(all_children_modules) == 6

    def test_no_child_model(self):
        class NoChildModel(nn.Module):
            def __init__(self):
                super(NoChildModel, self).__init__()

            def forward(self, x):
                pass

        no_child_model = NoChildModel()
        all_children_modules = get_all_children(no_child_model)
        assert len(all_children_modules) == 0


class Test_find_all_instances:
    def test_sequential(self):
        # Test case 1: Sequential model with nested Sequential modules
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Sequential(
                nn.Linear(20, 30),
                nn.ReLU(),
                nn.Linear(30, 40),
            ),
        )

        linear_layers = find_all_instances(model, nn.Linear)
        assert len(linear_layers) == 3

        relu_layers = find_all_instances(model, nn.ReLU)
        assert len(relu_layers) == 2

    def test_custom_model(self):
        # Test case 2: Custom model with nested modules
        class CustomModel(nn.Module):
            def __init__(self):
                super(CustomModel, self).__init__()
                self.layer1 = nn.Linear(10, 20)
                self.layer2 = nn.ReLU()
                self.submodule = nn.Sequential(
                    nn.Linear(20, 30),
                    nn.ReLU(),
                    nn.Linear(30, 40),
                )

            def forward(self, x):
                pass

        custom_model = CustomModel()

        linear_layers = find_all_instances(custom_model, nn.Linear)
        assert len(linear_layers) == 3

        relu_layers = find_all_instances(custom_model, nn.ReLU)
        assert len(relu_layers) == 2

    def test_no_child_model(self):
        # Test case 3: Model without any child module
        class NoChildModel(nn.Module):
            def __init__(self):
                super(NoChildModel, self).__init__()

            def forward(self, x):
                pass

        no_child_model = NoChildModel()

        linear_layers = find_all_instances(no_child_model, nn.Linear)
        assert len(linear_layers) == 0

        relu_layers = find_all_instances(no_child_model, nn.ReLU)
        assert len(relu_layers) == 0
