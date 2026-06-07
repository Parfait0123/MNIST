"""
MNISTClassifier — Architecture exacte issue du notebook d'entraînement.
  CNN Custom :
      Blocs Conv2d + BatchNorm2d + GELU + MaxPool2d
      Flatten -> LayerNorm -> Linear -> GELU -> Dropout -> Linear
"""

import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # --- BACKBONE CNN NATIF ---
        # Conçu pour des entrées de taille [Batch, 1, 32, 32]
        self.backbone = nn.Sequential(
            # Bloc Conv 1 : 32x32x1 -> 32x32x32 -> 16x16x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Bloc Conv 2 : 16x16x32 -> 16x16x64 -> 8x8x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),

            # Bloc Conv 3 : 8x8x64 -> 8x8x128 -> 6x6x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=0), # padding=0 réduit de 8x8 à 6x6
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.4),
            
            # Mise à plat automatique 
            nn.Flatten(start_dim=1)
        )

        num_features = 128 * 6 * 6

        #  TÊTE DE CLASSIFICATION ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image attendue : [Batch_Size, 1, 32, 32]
        x = self.backbone(image)
        return self.classifier(x)
