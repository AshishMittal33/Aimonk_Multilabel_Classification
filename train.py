import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import get_dataloader
from model import get_model
from dataset import MultiLabelDataset
from loss import compute_pos_weights, MaskedBCELoss


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (needed for pos_weight calculation)
    dataset = MultiLabelDataset("images", "labels.txt")

    # Compute imbalance weights
    pos_weight = compute_pos_weights(dataset).to(device)

    # Create dataloader
    dataloader = get_dataloader("images", "labels.txt", batch_size=16)

    # Load model
    model = get_model(num_attributes=4)
    model = model.to(device)

    # Loss
    criterion = MaskedBCELoss(pos_weight)

    # Optimizer (only final layer is trainable)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    num_epochs = 10
    loss_history = []
    iteration_number = 0

    model.train()

    for epoch in range(num_epochs):

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            iteration_number += 1

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

    # Plot loss curve
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("iteration_number")
    plt.ylabel("training_loss")
    plt.title("Aimonk_multilabel_problem")
    plt.savefig("loss_curve.png")
    plt.show()

    print("Loss curve saved as loss_curve.png")


if __name__ == "__main__":
    train()
