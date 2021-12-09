
import keras_tuner as kt

class ModelSpecification:
    name: str = None
    version: int = None
    builder = None
    description: str = None

    def __init__(self, name: str, version: int, builder, description: str = None):
        self.name = name
        self.version = version
        self.builder = builder
        self.description = description


_global_model_registry = {}


def register_model(name: str, version: int, builder, description: str = None):
    global _global_model_registry

    if name in _global_model_registry:
        assert version not in _global_model_registry[
            name], f"Model {name} of version {version} already exists in the registry"
    else:
        _global_model_registry[name] = {}

    _global_model_registry[name][version] = ModelSpecification(
        name, version, builder, description)


def get_model(name: str, version: int = None):
    global _global_model_registry

    if name in _global_model_registry:
        if version is None:
            most_recent_version = max(_global_model_registry[name].keys())
            return _global_model_registry[name][most_recent_version]
        elif version in _global_model_registry[name]:
            return _global_model_registry[name][version]

    return None


def get_model_builder(name: str, version: int = None, **kwargs):
    modelspec = get_model(name, version)

    if modelspec is not None:
        return lambda hp: modelspec.builder(hp, **kwargs)

    return None


def get_model_instance(name: str, version: int = None, **kwargs):
    modelspec = get_model(name, version)
    
    if modelspec is not None:
        return modelspec.builder(kt.HyperParameters(), **kwargs)

    return None
