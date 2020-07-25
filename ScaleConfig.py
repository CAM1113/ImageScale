import torch


class ScaleConfig:
    def __init__(self):
        self.input_shape = (512, 512)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


scale_config = ScaleConfig()
