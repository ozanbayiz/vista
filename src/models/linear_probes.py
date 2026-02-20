"""Linear probe classifiers for demographic attribute prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Independent linear heads for age, gender and race classification.

    Args:
        input_dim: Dimensionality of the shared input feature vector.
        age_classes: Number of age bins.
        gender_classes: Number of gender classes.
        race_classes: Number of race classes.
    """

    def __init__(self, input_dim: int, age_classes: int, gender_classes: int, race_classes: int) -> None:
        super().__init__()
        self.age_classifier = nn.Linear(input_dim, age_classes)
        self.gender_classifier = nn.Linear(input_dim, gender_classes)
        self.race_classifier = nn.Linear(input_dim, race_classes)

        self.input_dim = input_dim
        self.age_classes = age_classes
        self.gender_classes = gender_classes
        self.race_classes = race_classes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(age_logits, gender_logits, race_logits)``."""
        return (
            self.age_classifier(x),
            self.gender_classifier(x),
            self.race_classifier(x),
        )
