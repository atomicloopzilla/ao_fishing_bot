import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time

class FishingFloatDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load all image paths and labels
        for filename in os.listdir(data_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(data_dir, filename)
                self.image_paths.append(image_path)

                # Extract label from filename
                if '_bite_' in filename:
                    label = 1  # bite = positive class
                elif '_nobite_' in filename:
                    label = 0  # nobite = negative class
                else:
                    continue  # skip files with unknown format

                self.labels.append(label)

        print(f"Loaded {len(self.image_paths)} images")
        print(f"Bite samples: {sum(self.labels)}")
        print(f"No-bite samples: {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class FishingFloatClassifier(nn.Module):
    def __init__(self, input_size=100):
        super(FishingFloatClassifier, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 100x100 -> 50x50
            nn.Dropout2d(0.25),

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 50x50 -> 25x25
            nn.Dropout2d(0.25),

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 25x25 -> 12x12
            nn.Dropout2d(0.25),

            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 12x12 -> 6x6
            nn.Dropout2d(0.25),
        )

        # Calculate the size after conv layers
        # For 100x100 input: 100 -> 50 -> 25 -> 12 -> 6
        conv_output_size = 256 * 6 * 6

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Single output for binary classification
            nn.Sigmoid()  # Sigmoid for binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = model.to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_loss = float('inf')
    best_model_state = None

    print(f"\nStarting training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.float().to(device)

                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })

        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_acc = 100 * val_correct / val_total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)

        # Calculate additional metrics
        val_precision = precision_score(val_targets, val_predictions, zero_division=0)
        val_recall = recall_score(val_targets, val_predictions, zero_division=0)
        val_f1 = f1_score(val_targets, val_predictions, zero_division=0)

        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        # Learning rate scheduling
        scheduler.step(epoch_val_loss)

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            print(f'New best model saved (Val Loss: {best_val_loss:.4f})')

        print('-' * 60)

    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)')

    # Load best model
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader):
    """Evaluate model and show detailed metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()

            predicted = (outputs > 0.5).float()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probabilities.extend(outputs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)

    print(f'\nFinal Test Results:')
    print(f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Bite', 'Bite'],
                yticklabels=['No Bite', 'Bite'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return accuracy, precision, recall, f1

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Data directory
    data_dir = '../annotated_data'

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((100, 100)),  # Ensure all images are 100x100
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Custom transform for blur
    class RandomGaussianBlur:
        def __init__(self, probability=0.3, radius_range=(0.1, 2.0)):
            self.probability = probability
            self.radius_range = radius_range

        def __call__(self, img):
            if torch.rand(1) < self.probability:
                radius = torch.FloatTensor(1).uniform_(*self.radius_range).item()
                return img.filter(ImageFilter.GaussianBlur(radius=radius))
            return img

    # Custom transform for resolution downscaling
    class RandomDownscale:
        def __init__(self, probability=0.4, scale_range=(0.5, 0.8)):
            self.probability = probability
            self.scale_range = scale_range

        def __call__(self, img):
            if torch.rand(1) < self.probability:
                w, h = img.size
                scale = torch.FloatTensor(1).uniform_(*self.scale_range).item()
                new_w, new_h = int(w * scale), int(h * scale)
                # Downscale then upscale back to original size
                img = img.resize((new_w, new_h), Image.LANCZOS)
                img = img.resize((w, h), Image.LANCZOS)
            return img

    # Data augmentation for training with additional augmentations
    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(p=1.0),  # Flip every image
        RandomDownscale(probability=0.5, scale_range=(0.5, 0.8)),  # 50% resolution downscaling
        RandomGaussianBlur(probability=0.4, radius_range=(0.1, 2.0)),  # Add blur
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    full_dataset = FishingFloatDataset(data_dir, transform=transform)

    # Split dataset (80% train, 20% test)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply augmentation to training data
    train_dataset.dataset.transform = train_transform

    # Further split training data (80% train, 20% validation)
    train_size_actual = int(0.8 * train_size)
    val_size = train_size - train_size_actual

    train_dataset_final, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size_actual, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f'Training samples: {len(train_dataset_final)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')

    # Create model
    model = FishingFloatClassifier()
    print(f'\nModel created with {sum(p.numel() for p in model.parameters())} parameters')

    # Train model
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=50, learning_rate=0.001
    )

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)

    # Evaluate on test set
    accuracy, precision, recall, f1 = evaluate_model(trained_model, test_loader)

    # Save the trained model for PyTorch
    torch.save(trained_model.state_dict(), 'fishing_classifier_state_dict.pth')
    torch.save(trained_model, 'fishing_classifier_full_model.pth')

    # Convert to TorchScript for C++ LibTorch usage
    trained_model.eval()
    example_input = torch.randn(1, 3, 100, 100)  # Example input tensor

    # Method 1: Tracing (recommended for inference)
    traced_model = torch.jit.trace(trained_model, example_input)
    traced_model.save('fishing_classifier_traced.pt')

    # Method 2: Scripting (alternative method)
    try:
        scripted_model = torch.jit.script(trained_model)
        scripted_model.save('fishing_classifier_scripted.pt')
        print("Both traced and scripted models saved successfully!")
    except Exception as e:
        print(f"Scripting failed: {e}")
        print("Traced model saved successfully!")

    print(f'\nModel files saved:')
    print(f'- fishing_classifier_state_dict.pth (PyTorch state dict)')
    print(f'- fishing_classifier_full_model.pth (Full PyTorch model)')
    print(f'- fishing_classifier_traced.pt (TorchScript for C++)')

    # Save model summary
    with open('model_summary.txt', 'w') as f:
        f.write("Fishing Float Binary Classifier - Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {dataset_size} total images\n")
        f.write(f"Training: {len(train_dataset_final)} images\n")
        f.write(f"Validation: {len(val_dataset)} images\n")
        f.write(f"Test: {len(test_dataset)} images\n\n")
        f.write(f"Model Parameters: {sum(p.numel() for p in trained_model.parameters())}\n")
        f.write(f"Input Size: 100x100x3 (RGB)\n")
        f.write(f"Output: Single probability (0-1)\n")
        f.write(f"Classification Threshold: 0.5\n\n")
        f.write("Final Test Performance:\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Usage in C++:\n")
        f.write("Load the 'fishing_classifier_traced.pt' file using LibTorch\n")
        f.write("Input: Tensor of shape [1, 3, 100, 100] (normalized RGB image)\n")
        f.write("Output: Single float value (>0.5 = bite, <=0.5 = no bite)\n")

    print(f'\nTraining completed! Model summary saved to model_summary.txt')

if __name__ == '__main__':
    main()