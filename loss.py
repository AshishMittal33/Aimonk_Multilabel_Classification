import torch
import torch.nn as nn


def compute_pos_weights(dataset):
    """
    Compute positive weights for handling class imbalance.
    """

    labels = dataset.labels  # shape [N, 4]

    # Ignore NA (-1)
    valid_mask = labels != -1

    pos_counts = ((labels == 1) & valid_mask).sum(dim=0)
    neg_counts = ((labels == 0) & valid_mask).sum(dim=0)

    pos_weight = neg_counts / (pos_counts + 1e-6)

    return pos_weight


class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction='none'
        )

    def forward(self, outputs, targets):

        # Create mask for valid labels
        mask = targets != -1

        # Replace -1 with 0 (temporary, will be masked anyway)
        targets = torch.clamp(targets, min=0)

        loss_matrix = self.bce(outputs, targets)

        # Apply mask
        loss = (loss_matrix * mask).sum() / mask.sum()

        return loss
