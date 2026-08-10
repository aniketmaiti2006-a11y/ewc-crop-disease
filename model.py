import torch
import torch.nn as nn
import torchvision.models as models


# 1. Dynamically target the GPU if available (for cloud), otherwise fall back to CPU (for local testing)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. When you initialize your model instance later in the script:
#     model = CropDiseaseModel(...)
#     model = model.to(device)  # Forces the model layers into cloud GPU memory


RESNET50_FEATURE_DIM = 2048
  

class ResNet50Backbone(nn.Module):

    def __init__(self, pretrained: bool = True, freeze_early: bool = False):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        _resnet = models.resnet50(weights=weights)

        self.feature_extractor = nn.Sequential(*list(_resnet.children())[:-1])

        if freeze_early:
            for name, param in self.feature_extractor.named_parameters():
                if any(tag in name for tag in ["0.", "1.", "2.", "3.", "4.", "5."]):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        return feat.flatten(1)


class ClassificationHead(nn.Module):

    def __init__(
        self,
        in_features: int = RESNET50_FEATURE_DIM,
        num_classes: int = 10,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CropDiseaseModel(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_early: bool = False,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.backbone = ResNet50Backbone(
            pretrained=pretrained, freeze_early=freeze_early
        )
        self.head = ClassificationHead(
            in_features=RESNET50_FEATURE_DIM,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @torch.no_grad()
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def named_trainable_parameters(self):
        return (
            (n, p) for n, p in self.named_parameters() if p.requires_grad
        )
