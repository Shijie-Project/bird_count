from .shufflenet import get_shufflenet_density_model


def get_model(model_type, model_path, device, fuse):
    if model_type == "shufflenet":
        return get_shufflenet_density_model(model_path=model_path, device=device, fuse=fuse)
