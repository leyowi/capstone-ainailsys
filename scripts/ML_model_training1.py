import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import time
import copy
import json
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(r"D:\School\python\AINailSys")
DATA_DIR = PROJECT_ROOT / "data" / "04_stage1_binary"
MODEL_DIR = PROJECT_ROOT / "models" / "stage1_binary"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage1_results"

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
BATCH_SIZE = 32          # Number of images per batch
NUM_EPOCHS = 20          # Number of complete passes through dataset
LEARNING_RATE = 0.001    # How fast the model learns
IMAGE_SIZE = 224         # ResNet expects 224x224 images

# Device configuration (use GPU if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data_transforms():
    """
    Define image transformations for training and validation
    
    Training: Includes data augmentation
    Validation: Only basic preprocessing
    
    Returns:
        dict: Transforms for train and val
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),           # Flip 50% of images
            transforms.RandomRotation(10),                     # Rotate ±10 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust colors
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],       # ImageNet means
                               [0.229, 0.224, 0.225])          # ImageNet stds
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
    }
    
    return data_transforms


def load_data(data_dir, batch_size):
    """
    Load training and validation datasets
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for training
    
    Returns:
        tuple: (dataloaders, dataset_sizes, class_names)
    """
    data_transforms = get_data_transforms()
    
    # Load datasets
    image_datasets = {
        'train': datasets.ImageFolder(data_dir / 'train', data_transforms['train']),
        'val': datasets.ImageFolder(data_dir / 'val', data_transforms['val'])
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, 
                           shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, 
                         shuffle=False, num_workers=4)
    }
    
    # Get dataset info
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    return dataloaders, dataset_sizes, class_names


def create_model(num_classes):
    """
    Create ResNet18 model
    
    Args:
        num_classes: Number of output classes (2 for binary)
    
    Returns:
        model: ResNet18 model
    """
    # Load pre-trained ResNet18
    model = models.resnet18(pretrained=True)
    
    # Freeze early layers (optional - speeds up training)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Replace final layer for our task
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, 
                num_epochs, device, save_dir):
    """
    Train the model
    
    Args:
        model: Neural network model
        dataloaders: Train and val dataloaders
        dataset_sizes: Sizes of train and val datasets
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
        save_dir: Directory to save model
    
    Returns:
        tuple: (trained_model, history)
    """
    since = time.time()
    
    # Track best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Track history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    print(f"\nDevice: {device}")
    print(f"Train size: {dataset_sizes['train']}")
    print(f"Val size: {dataset_sizes['val']}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}\n")
    print("=" * 80)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 80)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            batch_count = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                batch_count += 1
                
                # Show progress every 10 batches
                if batch_count % 10 == 0:
                    batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                    print(f'   {phase} - Batch {batch_count}: Loss={loss.item():.4f}, Acc={batch_acc:.4f}')
            
            # Calculate epoch statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Save to history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'\n{phase.upper()} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
            
            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save checkpoint
                checkpoint_path = save_dir / 'checkpoints' / f'epoch_{epoch+1}_acc_{epoch_acc:.4f}.pth'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_acc,
                }, checkpoint_path)
                
                print(f'    New best model saved! Acc: {epoch_acc:.4f}')
    
    # Training complete
    time_elapsed = time.time() - since
    print('\n' + "=" * 80)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    print("=" * 80)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


def plot_training_history(history, save_path):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Training history dictionary
        save_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n Training curves saved to: {save_path}')
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print(" AINAILSYS - STAGE 1 MODEL TRAINING")
    print("   Binary Classification: Healthy vs Anemic")
    print("=" * 80)
    print()
    
    # Check GPU
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Load data
    print(" Loading data...")
    dataloaders, dataset_sizes, class_names = load_data(DATA_DIR, BATCH_SIZE)
    print(f"   Classes: {class_names}")
    print(f"   Train: {dataset_sizes['train']} images")
    print(f"   Val: {dataset_sizes['val']} images")
    print()
    
    # Create model
    print("  Creating model...")
    model = create_model(num_classes=len(class_names))
    model = model.to(DEVICE)
    print(f"   Model: ResNet18")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_dir=MODEL_DIR
    )
    
    # Save final model
    final_model_path = MODEL_DIR / 'best_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'\n Final model saved to: {final_model_path}')
    
    # Save training history
    history_path = OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f' Training history saved to: {history_path}')
    
    # Plot training curves
    plot_path = OUTPUT_DIR / 'training_curves.png'
    plot_training_history(history, plot_path)
    
    # Save training info
    info = {
        'model': 'ResNet18',
        'task': 'Binary Classification (Healthy vs Anemic)',
        'classes': class_names,
        'epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'device': str(DEVICE),
        'train_size': dataset_sizes['train'],
        'val_size': dataset_sizes['val'],
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc'])
    }
    
    info_path = OUTPUT_DIR / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f' Training info saved to: {info_path}')
    
    # Final summary
    print("\n" + "=" * 80)
    print(" STAGE 1 TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n Final Results:")
    print(f"   Best Validation Accuracy: {max(history['val_acc']):.4f} ({max(history['val_acc'])*100:.2f}%)")
    print(f"   Final Train Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"   Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    
    print(f"\n Saved Files:")
    print(f"   Model: {final_model_path}")
    print(f"   History: {history_path}")
    print(f"   Curves: {plot_path}")
    print(f"   Info: {info_path}")
    
    
    print("\n" + "=" * 80)