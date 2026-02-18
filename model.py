import torch
import torch.nn as nn
import torchvision.models as models


def get_model(num_attributes=4):

    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_attributes)

    return model
