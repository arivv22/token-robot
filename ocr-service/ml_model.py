import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

# -----------------------------
# 🧠 CUSTOM DATASET CLASS
# -----------------------------
class TokenDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Preprocess similar to main OCR
        image = self.crop_token_area(image)
        
        if self.transform:
            image = self.transform(image)
            
        # Convert label to tensor (20-digit token as sequence of digits)
        label = self.labels[idx]
        label_tensor = torch.tensor([int(d) for d in label], dtype=torch.long)
        
        return image, label_tensor
    
    def crop_token_area(self, img):
        """Same cropping logic as main OCR"""
        width, height = img.size
        
        left = int(width * 0.25)
        top = int(height * 0.35)
        right = int(width * 0.75)
        bottom = int(height * 0.65)
        
        return img.crop((left, top, right, bottom))

# -----------------------------
# 🏗️ CNN MODEL ARCHITECTURE
# -----------------------------
class TokenOCRModel(nn.Module):
    def __init__(self, num_classes=10, sequence_length=20):
        super(TokenOCRModel, self).__init__()
        
        # CNN layers for feature extraction
        self.conv_layers = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(
            input_size=256 * 4 * 4,  # flattened CNN output
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Output layer for each digit position
        self.output_layer = nn.Linear(512 * 2, num_classes)  # bidirectional LSTM
        
        self.sequence_length = sequence_length
        
    def forward(self, x):
        # CNN feature extraction
        batch_size = x.size(0)
        conv_features = self.conv_layers(x)
        
        # Flatten for LSTM
        conv_flat = conv_features.view(batch_size, 1, -1)  # (batch, seq_len=1, features)
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_flat)
        
        # Expand to sequence length
        lstm_expanded = lstm_out.expand(-1, self.sequence_length, -1)
        
        # Output for each digit position
        outputs = self.output_layer(lstm_expanded)
        
        return outputs  # (batch, sequence_length, num_classes)

# -----------------------------
# 🎯 TRAINING PIPELINE
# -----------------------------
class TokenOCRTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss for each digit position
            loss = 0
            for i in range(labels.size(1)):  # for each digit position
                loss += self.criterion(outputs[:, i, :], labels[:, i])
            
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 2)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.numel()
            
            total_loss += loss.item()
            
            # Update progress bar
            current_acc = correct_predictions / total_predictions
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                
                # Calculate loss for each digit position
                loss = 0
                for i in range(labels.size(1)):
                    loss += self.criterion(outputs[:, i, :], labels[:, i])
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 2)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.numel()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, save_path='models/best_model.pth'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, save_path)
                print(f"✅ New best model saved! Val Loss: {val_loss:.4f}")
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies
    
    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            if isinstance(image, str):
                # Load image from path
                img = Image.open(image).convert('RGB')
                img = self.crop_token_area(img)
                transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = transform(img).unsqueeze(0).to(self.device)
            
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 2)
            
            # Convert to string
            token = ''.join([str(d.item()) for d in predicted[0]])
            
            return token
    
    def crop_token_area(self, img):
        """Same cropping logic as main OCR"""
        width, height = img.size
        
        left = int(width * 0.25)
        top = int(height * 0.35)
        right = int(width * 0.75)
        bottom = int(height * 0.65)
        
        return img.crop((left, top, right, bottom))

# -----------------------------
# 📊 EVALUATION METRICS
# -----------------------------
class ModelEvaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
    
    def evaluate(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 2)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Overall accuracy
        overall_accuracy = np.mean(all_predictions == all_labels)
        
        # Per-digit accuracy
        digit_accuracies = []
        for i in range(all_labels.shape[1]):  # for each digit position
            digit_acc = np.mean(all_predictions[:, i] == all_labels[:, i])
            digit_accuracies.append(digit_acc)
        
        # Complete token accuracy (all 20 digits correct)
        complete_tokens_correct = np.all(all_predictions == all_labels, axis=1)
        complete_token_accuracy = np.mean(complete_tokens_correct)
        
        # Character-level metrics
        y_true = all_labels.flatten()
        y_pred = all_predictions.flatten()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        results = {
            'overall_accuracy': overall_accuracy,
            'complete_token_accuracy': complete_token_accuracy,
            'digit_accuracies': digit_accuracies,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': len(all_labels)
        }
        
        return results
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs, save_path='plots/training_history.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(train_accs, label='Train Accuracy', color='blue')
        ax2.plot(val_accs, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training history plot saved to {save_path}")

# -----------------------------
# 📁 DATA PREPARATION
# -----------------------------
def prepare_data(data_dir, test_size=0.2, val_size=0.2):
    """
    Prepare training data from directory structure:
    data_dir/
        ├── image1.jpg
        ├── image2.jpg
        └── labels.json  # {"image1.jpg": "22592297522702336675", ...}
    """
    
    # Load labels
    labels_file = os.path.join(data_dir, 'labels.json')
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        # Create dummy labels for testing
        image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        labels = {f: "22592297522702336675" for f in image_files}
    
    # Prepare image paths and labels
    image_paths = []
    image_labels = []
    
    for img_file, label in labels.items():
        img_path = os.path.join(data_dir, img_file)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            image_labels.append(label)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, image_labels, test_size=test_size, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_transforms():
    """Data augmentation and preprocessing transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

# -----------------------------
# 🚀 MAIN TRAINING SCRIPT
# -----------------------------
def main():
    # Configuration
    DATA_DIR = 'data/training'
    MODEL_SAVE_PATH = 'models/token_ocr_model.pth'
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    print("🚀 Starting Token OCR Training Pipeline")
    print("=" * 50)
    
    # Prepare data
    print("📁 Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(DATA_DIR)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Get transforms
    train_transform, val_test_transform = get_transforms()
    
    # Create datasets
    train_dataset = TokenDataset(X_train, y_train, train_transform)
    val_dataset = TokenDataset(X_val, y_val, val_test_transform)
    test_dataset = TokenDataset(X_test, y_test, val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("🧠 Initializing model...")
    model = TokenOCRModel(num_classes=10, sequence_length=20)
    
    # Initialize trainer
    trainer = TokenOCRTrainer(model)
    
    # Train model
    print("🏋️ Starting training...")
    train_losses, val_losses, train_accs, val_accs = trainer.train(
        train_loader, val_loader, epochs=EPOCHS, save_path=MODEL_SAVE_PATH
    )
    
    # Evaluate model
    print("📊 Evaluating model...")
    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(test_loader)
    
    # Print results
    print("\n🎯 Evaluation Results:")
    print("=" * 30)
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Complete Token Accuracy: {results['complete_token_accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Total Samples: {results['total_samples']}")
    
    print("\n📈 Per-digit Accuracies:")
    for i, acc in enumerate(results['digit_accuracies']):
        print(f"Digit {i+1}: {acc:.4f}")
    
    # Plot training history
    evaluator.plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Save results
    with open('models/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Training completed successfully!")
    print(f"📁 Model saved to: {MODEL_SAVE_PATH}")
    print(f"📊 Results saved to: models/evaluation_results.json")

if __name__ == "__main__":
    main()
