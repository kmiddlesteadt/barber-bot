"""
Transfer learning with ResNet50 for face-shape classification.

Expected dataset structure:
src/training/face_shapes/
    Oval/
    Square/
    Round/
    Heart/
    Diamond/
"""

import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, random_split

from vision_service import VisionService


def create_resnet50_model(num_classes=5, pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze earlier layers and fine-tune the deeper feature blocks plus final head.
    for name, param in model.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1", "layer2")):
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def create_synthetic_face_loaders(batch_size=32):
    """Small fallback loaders so the script can run before a real dataset exists."""
    n_train = 200
    n_val = 50
    X_train = torch.randn(n_train, 3, 224, 224)
    y_train = torch.randint(0, len(VisionService.FACE_SHAPES), (n_train,))
    X_val = torch.randn(n_val, 3, 224, 224)
    y_val = torch.randint(0, len(VisionService.FACE_SHAPES), (n_val,))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def build_data_loaders(data_dir, batch_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if not os.path.exists(data_dir):
        print(f"Face-shape dataset not found: {data_dir}")
        print("Using synthetic tensors for a smoke-test training run.")
        return create_synthetic_face_loaders(batch_size)

    dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    print(f"Loaded {len(dataset)} images from {data_dir}")
    print(f"Classes: {dataset.classes}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def save_resnet_model(model, epoch, accuracy):
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "resnet50_face_shape.pth")
    torch.save(model.state_dict(), model_path)

    mapping_path = os.path.join(model_dir, "face_shape_classes.pkl")
    with open(mapping_path, "wb") as mapping_file:
        pickle.dump(VisionService.FACE_SHAPES, mapping_file)

    print(f"Model saved to {model_path}")
    print(f"Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}%")


def train_resnet_model(data_dir=None, epochs=20, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "training", "face_shapes")

    train_loader, val_loader = build_data_loaders(data_dir, batch_size)

    print("Creating ResNet50 model with transfer learning...")
    model = create_resnet50_model(num_classes=len(VisionService.FACE_SHAPES), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / max(train_total, 1)
        val_acc = 100 * val_correct / max(val_total, 1)

        print(
            f"Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {train_loss / max(len(train_loader), 1):.4f}, "
            f"Train Acc: {train_acc:.2f}%, "
            f"Val Loss: {val_loss / max(len(val_loader), 1):.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_resnet_model(model, epoch, val_acc)

        scheduler.step()

    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    return model


def main():
    print("=" * 60)
    print("ResNet50 Face Shape Classifier - Transfer Learning")
    print("=" * 60)
    train_resnet_model(epochs=20, batch_size=32, learning_rate=0.001)
    print("Model training and saving complete.")


if __name__ == "__main__":
    main()
