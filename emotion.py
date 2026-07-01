import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import time


class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Block 2: (Input: 24x24x64) -> (Output: 12x12x128)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        # Block 3: (Input: 12x12x128) -> (Output: 6x6x256)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Block 4: (Input: 6x6x256) -> (Output: 3x3x512)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        
        # Classifier Head: (Input: 3*3*512 = 4608)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Connect the blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x

# --- FUNCTION TO CONTAIN ALL RUNNING CODE ---
def main():
    # --- 1. SETUP PARAMETERS AND DEVICE ---
    print("Setting up parameters...")
    
    TRAIN_DIR = 'train'
    VALID_DIR = 'test'
    IMG_SIZE = 48
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. DATA TRANSFORMS AND LOADING ---
    print("Setting up data transforms and loaders...")
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48,48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]),
    }

    print(f"Loading data from {TRAIN_DIR} and {VALID_DIR}...")
    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, transform=data_transforms['train']),
        'val': datasets.ImageFolder(VALID_DIR, transform=data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")
    print(f"Training samples: {dataset_sizes['train']}, Validation samples: {dataset_sizes['val']}")

    # --- 3. HANDLE CLASS IMBALANCE (THE BIAS) ---
    print("Calculating class weights...")
    train_labels = [label for _, label in image_datasets['train'].samples]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Class weights calculated:")
    for i, label in enumerate(class_names):
        print(f"  {label.capitalize()}: {class_weights[i]:.2f}")

    # --- 4. CREATE MODEL INSTANCE ---
    print("Building the PyTorch model...")
    model = EmotionCNN(num_classes)
    model = model.to(device)
    print(f"Model built successfully and moved to {device}.")

    # --- 5. INITIALIZE LOSS AND OPTIMIZER ---
    print("Initializing loss function (with weights) and optimizer...")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 6. THE TRAINING LOOP ---
    print("\nStarting Training...")
    print("-" * 70)
    
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f'Epoch {epoch+1}/{EPOCHS}')
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / dataset_sizes['val']
        epoch_val_acc = val_corrects.double() / dataset_sizes['val']
        
        epoch_time = time.time() - start_time
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | Time: {epoch_time:.0f}s')

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'bestModelOnCleanDataset_1.pth')
            print(f"  New best model saved with accuracy: {best_val_acc:.4f}")

    print("-" * 70)
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")
    print("Best model saved as 'bestModelOnCleanDataset_1.pth'")

if __name__ == '__main__':
    main()
