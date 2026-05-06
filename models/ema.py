from copy import deepcopy

import torch


class ModelEMA:
    """Exponential Moving Average for Model Parameters"""

    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(model_v * (1.0 - self.decay) + ema_v * self.decay)
