import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ImageNetRenormalize(nn.Module):
    """Renormalize images from [-1, 1] to ImageNet"""

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, orig_min=-1, orig_max=1):
        super().__init__()
        self.orig_min = orig_min
        self.orig_max = orig_max
        self.mean = torch.tensor(self.MEAN).view(1, -1, 1, 1)
        self.std = torch.tensor(self.STD).view(1, -1, 1, 1)

    def forward(self, x):
        x01 = (x - self.orig_min) / (self.orig_max - self.orig_min)
        return (x01 - self.mean.to(x01.device)) / self.std.to(x01.device)


class DenseResBlock(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.impl = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels, elementwise_affine=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        return x + self.impl(x)


class MLPApproximator(nn.Module):
    def __init__(self, w_size=512, dz_size=3, depth=8):
        super().__init__()
        inner_size = w_size + dz_size
        self.impl = nn.Sequential(*[DenseResBlock(inner_size) for _ in range(depth)])
        self.out = nn.Linear(inner_size, w_size)

    def forward(self, x):
        return self.out(self.impl(x))


class ResNetEncoder(nn.Module):
    def __init__(self, num_levels=7, w_size=512,  w_bottleneck_size=2048):
        super().__init__()
        self.num_levels = num_levels
        self.renorm = ImageNetRenormalize()
        self.backbone = torchvision.models.resnet152(pretrained=True)

        self.backbone.fc = nn.Sequential(
            nn.Linear(self.backbone.fc.in_features + 1280, w_bottleneck_size),
            nn.LayerNorm(w_bottleneck_size),
            nn.LeakyReLU(0.2),
            nn.Linear(w_bottleneck_size, w_size * num_levels)
        )
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool3 = nn.AdaptiveAvgPool2d(1)

    def _predict(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        pooled1 = torch.flatten(self.avgpool1(x), 1)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        pooled3 = torch.flatten(self.avgpool3(x), 1)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((pooled1, pooled3, x), dim=-1)
        x = self.backbone.fc(x)
        return x

    def forward(self, image):
        if image.dim() == 5:
            image = image.squeeze(1)
            need_unsqueeze = True
        else:
            need_unsqueeze = False

        image = self.renorm(image)
        predicts = self._predict(image)
        w_by_layer = predicts.chunk(self.num_levels, 1)
        result = {f'latent_wprime:{layer_i}:{j}': w_by_layer[layer_i].contiguous()
                  for layer_i in range(self.num_levels)
                  for j in range(2)}

        if need_unsqueeze:
            for name in list(result):
                result[name] = result[name].unsqueeze(1)

        return result
