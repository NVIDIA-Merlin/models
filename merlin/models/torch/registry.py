from merlin.models.utils.registry import Registry, RegistryMixin

registry: Registry = Registry.class_registry("modules")


class TorchRegistryMixin(RegistryMixin):
    registry = registry


__all__ = ["registry", "Registry", "RegistryMixin", "TorchRegistryMixin"]
