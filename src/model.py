import torch
import torch.nn as nn
import numpy as np
from typing import Union

class ResidualBlock(nn.Module):
    """
    Residual Block for tabular deep learning.
    Allows deeper gradient propagation and avoids degradation.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))

class TitanicDeepNet(nn.Module):
    """
    Deep Neural Network for Titanic Passenger Survival Classification.
    Combines Batch Normalization, Residual Connections, SiLU activations,
    and Dropout regularization to maximize tabular generalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim

        # Input projection layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )

        # Residual processing block
        self.res_block = ResidualBlock(hidden_dim, dropout_rate=dropout_rate)

        # Dense reduction layers
        self.dense_block = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.75)
        )

        # Output logit projection
        self.output_head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        # Handle single sample during inference by ensuring batch dimension
        is_single = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            is_single = True

        out = self.input_layer(x)
        out = self.res_block(out)
        out = self.dense_block(out)
        logits = self.output_head(out)

        if is_single:
            logits = logits.squeeze(0)
        return logits

    def predict_proba(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Inference helper returning survival probability between [0, 1]."""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            return probs

    def predict(self, x: Union[torch.Tensor, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Inference helper returning binary survival prediction (0 or 1)."""
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(int)
