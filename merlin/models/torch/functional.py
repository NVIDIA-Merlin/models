import builtins
import inspect
from copy import deepcopy
from functools import wraps
from typing import Callable, Iterable, Protocol, Tuple, TypeVar, Union, runtime_checkable

from torch import nn


@runtime_checkable
class HasBool(Protocol):
    def __bool__(self) -> bool:
        ...


_TModule = TypeVar("_TModule", bound=nn.Module)
ModuleFunc = Callable[[nn.Module], nn.Module]
ModuleIFunc = Callable[[nn.Module, int], nn.Module]
ModulePredicate = Callable[[nn.Module], Union[bool, HasBool]]
ModuleMapFunc = Callable[[nn.Module], Union[nn.Module, None]]


class ContainerMixin:
    """Mixin that can be used to give a class container-like behavior.

    This mixin provides a set of methods that come from functional programming.

    """

    def filter(self: _TModule, func: ModulePredicate, recurse: bool = False) -> _TModule:
        """
        Returns a new container with modules that satisfy the filtering function.

        Example usage::
            >>> block = Block(nn.LazyLinear(10))
            >>> block.filter(lambda module: isinstance(module, nn.Linear))
            Block(nn.Linear(10, 10))

        Parameters
        ----------
        func (Callable[[Module], bool]): A function that takes a module and returns
            a boolean or a boolean-like object.
        recurse (bool, optional): Whether to recursively filter modules
            within sub-containers. Default is False.

        Returns
        -------
            Self: A new container with the filtered modules.
        """

        _to_call = _recurse(func, "filter") if recurse else func
        output = self.__class__()

        for module in self:
            filtered = _to_call(module)
            if filtered:
                if isinstance(filtered, bool):
                    output.append(module)
                else:
                    output.append(filtered)

        return output

    def flatmap(self: _TModule, func: ModuleFunc) -> _TModule:
        """
        Applies a function to each module and flattens the results into a new container.

        Example usage::
            >>> block = Block(nn.LazyLinear(10))
            >>> container.flatmap(lambda module: [module, module])
            Block(nn.LazyLinear(10), nn.LazyLinear(10))

        Parameters
        ----------
        func : Callable[[Module], Iterable[Module]]
            A function that takes a module and returns an iterable of modules.

        Returns
        -------
        Self
            A new container with the flattened modules.

        Raises
        ------
        TypeError
            If the input function is not callable.
        RuntimeError
            If an exception occurs during mapping the function over the module.
        """

        if not callable(func):
            raise TypeError(f"Expected callable function, received: {type(func).__name__}")

        try:
            mapped = self.map(func)
        except Exception as e:
            raise RuntimeError("Failed to map function over the module") from e

        output = self.__class__()

        try:
            for sublist in mapped:
                for item in sublist:
                    output.append(item)
        except TypeError as e:
            raise TypeError("Function did not return an iterable object") from e

        return output

    def forall(self, func: ModulePredicate, recurse: bool = False) -> bool:
        """
        Checks if the given predicate holds for all modules in the container.

        Example usage::
            >>> block = Block(nn.LazyLinear(10))
            >>> container.forall(lambda module: isinstance(module, nn.Module))
            True

        Parameters
        ----------
        func : Callable[[Module], bool]
            A predicate function that takes a module and returns True or False.
        recurse : bool, optional
            Whether to recursively check modules within sub-containers. Default is False.

        Returns
        -------
        bool
            True if the predicate holds for all modules, False otherwise.


        """
        _to_call = _recurse(func, "forall") if recurse else func
        return all(_to_call(module) for module in self)

    def map(self: _TModule, func: ModuleFunc, recurse: bool = False) -> _TModule:
        """
        Applies a function to each module and returns a new container with the results.

        Example usage::
            >>> block = Block(nn.LazyLinear(10))
            >>> container.map(lambda module: nn.ReLU())
            Block(nn.ReLU())

        Parameters
        ----------
        func : Callable[[Module], Module]
            A function that takes a module and returns a modified module.
        recurse : bool, optional
            Whether to recursively map the function to modules within sub-containers.
            Default is False.

        Returns
        -------
        _TModule
            A new container with the mapped modules.
        """

        _to_call = _recurse(func, "map") if recurse else func

        return self.__class__(*(_to_call(module) for module in self))

    def mapi(self: _TModule, func: ModuleIFunc, recurse: bool = False) -> _TModule:
        """
        Applies a function to each module along with its index and
        returns a new container with the results.

        Example usage::
            >>> block = Block(nn.LazyLinear(10), nn.LazyLinear(10))
            >>> container.mapi(lambda module, i: module if i % 2 == 0 else nn.ReLU())
            Block(nn.LazyLinear(10), nn.ReLU())

        Parameters
        ----------
        func : Callable[[Module, int], Module]
            A function that takes a module and its index,
            and returns a modified module.
        recurse : bool, optional
            Whether to recursively map the function to modules within
            sub-containers. Default is False.

        Returns
        -------
        Self
            A new container with the mapped modules.
        """

        _to_call = _recurse(func, "mapi") if recurse else func
        return self.__class__(*(_to_call(module, i) for i, module in enumerate(self)))

    def choose(self: _TModule, func: ModuleMapFunc, recurse: bool = False) -> _TModule:
        """
        Returns a new container with modules that are selected by the given function.

        Example usage::
            >>> block = Block(nn.LazyLinear(10), nn.Relu())
            >>> container.choose(lambda m: m if isinstance(m, nn.Linear) else None)
            Block(nn.LazyLinear(10))

        Parameters
        ----------
        func : Callable[[Module], Union[Module, None]]
            A function that takes a module and returns a module or None.
        recurse : bool, optional
            Whether to recursively choose modules within sub-containers. Default is False.

        Returns
        -------
        Self
            A new container with the chosen modules.
        """

        to_add = []
        _to_call = _recurse(func, "choose") if recurse else func

        for module in self:
            f_out = _to_call(module)
            if f_out:
                to_add.append(f_out)

        return self.__class__(*to_add)

    def walk(self: _TModule, func: ModulePredicate) -> _TModule:
        """
        Applies a function to each module recursively and returns
        a new container with the results.

        Example usage::
            >>> block = Block(Block(nn.LazyLinear(10), nn.ReLU()))
            >>> block.walk(lambda m: m if isinstance(m, nn.ReLU) else None)
            Block(Block(nn.ReLU()))

        Parameters
        ----------
        func : Callable[[Module], Module]
            A function that takes a module and returns a modified module.

        Returns
        -------
        Self
            A new container with the walked modules.
        """

        return self.map(func, recurse=True)

    def zip(self, other: Iterable[_TModule]) -> Iterable[Tuple[_TModule, _TModule]]:
        """
        Zips the modules of the container with the modules from another iterable into pairs.

        Example usage::
            >>> list(Block(nn.Linear(10)).zip(Block(nn.Linear(10))))
            [(nn.Linear(10), nn.Linear(10))]

        Parameters
        ----------
        other : Iterable[Self]
            Another iterable containing modules to be zipped with.

        Returns
        -------
        Iterable[Tuple[Self, Self]]
            An iterable of pairs containing modules from the container
            and the other iterable.
        """

        return builtins.zip(self, other)

    def freeze(self) -> None:
        """
        Freezes the parameters of all modules in the container
        by setting `requires_grad` to False.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """
        Unfreezes the parameters of all modules in the container
        by setting `requires_grad` to True.
        """
        for param in self.parameters():
            param.requires_grad = True

    def __add__(self, module) -> _TModule:
        if hasattr(module, "__iter__"):
            return self.__class__(*self, *module)

        return self.__class__(*self, module)

    def __radd__(self, module) -> _TModule:
        if hasattr(module, "__iter__"):
            return self.__class__(*module, *self)

        return self.__class__(module, *self)


def map(
    module: _TModule,
    func: ModuleFunc,
    recurse: bool = False,
    parameterless_modules_only=False,
    **kwargs,
) -> _TModule:
    """
    Applies a transformation function to a module or a collection of modules.

    Parameters
    ----------
    module : nn.Module
        The module or collection of modules to which the function will be applied.
    func : ModuleFunc
        The function that will be applied to the modules.
    recurse : bool, optional
        Whether to apply the function recursively to child modules.
    parameterless_modules_only : bool, optional
        Whether to apply the function only to modules without parameters.
    **kwargs : dict
        Additional keyword arguments that will be passed to the transformation function.

    Returns
    -------
    type(module)
        The transformed module or collection of modules.
    """
    if hasattr(module, "map"):
        to_call = module.map
    elif isinstance(module, Iterable):
        # Check if the module has .items() method (for dict-like modules)
        if hasattr(module, "items"):
            to_call = map_module_dict
        else:
            to_call = map_module_list
    else:
        to_call = map_module

    return to_call(
        module,
        func,
        parameterless_modules_only=parameterless_modules_only,
        recurse=recurse,
        **kwargs,
    )


def walk(
    module: _TModule, func: ModuleFunc, parameterless_modules_only=False, **kwargs
) -> _TModule:
    """
    Applies a transformation function recursively to a module or a collection of modules.

    Parameters
    ----------
    module : nn.Module
        The module or collection of modules to which the function will be applied.
    func : ModuleFunc
        The function that will be applied to the modules.
    parameterless_modules_only : bool, optional
        Whether to apply the function only to modules without parameters.
    **kwargs : dict
        Additional keyword arguments that will be passed to the transformation function.

    Returns
    -------
    type(module)
        The transformed module or collection of modules.
    """
    return map(
        module, func, recurse=True, parameterless_modules_only=parameterless_modules_only, **kwargs
    )


def map_module(
    module: _TModule, func: ModuleFunc, recurse=False, parameterless_modules_only=False, **kwargs
) -> _TModule:
    """
    Applies a transformation function to a module and optionally to its child modules.

    Parameters
    ----------
    module : nn.Module
        The module to which the function will be applied.
    func : ModuleFunc
        The function that will be applied to the module.
    recurse : bool, optional
        Whether to apply the function recursively to child modules.
    parameterless_modules_only : bool, optional
        Whether to apply the function only to modules without parameters.
    **kwargs : dict
        Additional keyword arguments that will be passed to the transformation function.

    Returns
    -------
    nn.Module
        The transformed module.
    """
    if list(module.parameters(recurse=False)):
        new_module = module
    else:
        new_module = deepcopy(module)

    f_kwargs = _get_func_kwargs(func, **kwargs)
    new_module = func(new_module, **f_kwargs)

    if new_module is not module and recurse:
        for i, (name, child) in enumerate(module.named_children()):
            setattr(
                new_module,
                name,
                map(child, func, recurse, parameterless_modules_only, i=i, name=name),
            )

    return new_module


def map_module_list(
    module_list: _TModule, func, recurse=False, parameterless_modules_only=False, **kwargs
) -> _TModule:
    mapped_modules = []
    for i, module in enumerate(module_list):
        new_module = map(
            module,
            func,
            recurse=recurse,
            parameterless_modules_only=parameterless_modules_only,
            i=i,
            name=str(i),
            **kwargs,
        )
        mapped_modules.append(new_module)

    return _create_list_wrapper(module_list, mapped_modules)


def map_module_dict(
    module_dict: _TModule,
    func: ModuleFunc,
    recurse: bool = False,
    parameterless_modules_only: bool = False,
    **kwargs,
) -> _TModule:
    """
    Applies a transformation function to a ModuleDict of modules.

    Parameters
    ----------
    module_dict : nn.ModuleDict
        The ModuleDict of modules to which the function will be applied.
    func : ModuleFunc
        The function that will be applied to the modules.
    recurse : bool, optional
        Whether to apply the function recursively to child modules.
    parameterless_modules_only : bool, optional
        Whether to apply the function only to modules without parameters.
    **kwargs : dict
        Additional keyword arguments that will be passed to the transformation function.

    Returns
    -------
    nn.ModuleDict
        The ModuleDict of transformed modules.
    """

    # Map the function to each module in the dictionary
    mapped_modules = {}
    for i, (name, module) in enumerate(module_dict.items()):
        mapped_modules[name] = map(
            module,
            func,
            recurse=recurse,
            parameterless_modules_only=parameterless_modules_only,
            name=name,
            i=i,
            **kwargs,
        )

    return type(module_dict)(mapped_modules)


def _create_list_wrapper(module_list, to_add):
    # Check the signature of the type constructor
    sig = inspect.signature(type(module_list).__init__)
    if "args" in sig.parameters:
        return type(module_list)(*to_add)  # Unpack new_modules

    return type(module_list)(to_add)  # Don't unpack new_modules


def _get_func_kwargs(func, **kwargs):
    sig = inspect.signature(func)
    f_kwargs = {}
    if "i" in sig.parameters and "i" in kwargs:
        f_kwargs["i"] = kwargs["i"]
    if "name" in sig.parameters and "name" in kwargs:
        f_kwargs["name"] = kwargs["name"]

    return f_kwargs


def _recurse(func, to_recurse_name: str):
    @wraps(func)
    def inner(module, *args, **kwargs):
        if hasattr(module, to_recurse_name):
            fn = getattr(module, to_recurse_name)

            return fn(func, *args, **kwargs)

        return func(module, *args, **kwargs)

    return inner
