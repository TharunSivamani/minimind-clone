import torch
import torch.nn as nn
import torch.optim as optim


class LoRA(nn.Module):

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        # Gaussian Initialization of Matrix A
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # Matrix B initialized with all zero's
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank: int = 8):

    """
    Applies LoRA to every Linear layer in the model whose output_features == in_features
    """

    # device = next(model.parameters()).device

    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Linear)
            and module.weight.shape[0] == module.weight.shape[1]
        ):
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(
                model.device
            )
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):

    # device = next(model.parameters()).device
    
    state_dict = torch.load(path, map_location=model.device)
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora.", ""): v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                f"{name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)

    torch.save(state_dict, path)
