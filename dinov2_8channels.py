import torch
import torch.nn as nn

def load_dinov2_vitl14():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    return model

def replicate_layer(model_layer):

    original_layer = model_layer

    # Extract weights from the original layer
    original_weights = original_layer.weight.data

    # Replicate the weights across the channel dimension (3 times)
    replicated_weights = torch.repeat_interleave(original_weights, 3, dim=1)
    replicated_weights = replicated_weights[:, :8, :, :]
    
    # copy weights
    new_layer = nn.Conv2d(8, original_layer.out_channels, kernel_size=original_layer.kernel_size, stride=original_layer.stride)


    new_layer.weight.data = replicated_weights

    if original_layer.bias is not None:
        new_layer.bias.data = original_layer.bias.data

    return new_layer


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, model):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = model

        # dinov2-s14
        # self.classifier = nn.Sequential(
        #     nn.Linear(384, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2)
        # )

        # dinov2-vit l 14
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x
