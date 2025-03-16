import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from load import DataCC
from model import ImprovedCNN

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (images, (steering_targets, throttle_targets)) in enumerate(train_loader):
            images = images.to(device)
            steering_targets = steering_targets.to(device)
            throttle_targets = throttle_targets.to(device)
            
            optimizer.zero_grad()
            # AMP autocasting
            with torch.cuda.amp.autocast():
                steering_pred, throttle_pred = model(images)
                loss_steering = criterion(steering_pred, steering_targets)
                loss_throttle = criterion(throttle_pred, throttle_targets)
                loss = loss_steering + loss_throttle
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, (steering_targets, throttle_targets) in val_loader:
                images = images.to(device)
                steering_targets = steering_targets.to(device)
                throttle_targets = throttle_targets.to(device)
                
                with torch.cuda.amp.autocast():
                    steering_pred, throttle_pred = model(images)
                    loss_steering = criterion(steering_pred, steering_targets)
                    loss_throttle = criterion(throttle_pred, throttle_targets)
                    loss = loss_steering + loss_throttle
                    
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset_root = "dc_dataset_2"
    dataset = DataCC(root_dir=dataset_root, transform=transform, test=False)
    
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = ImprovedCNN().to(device)
    
    num_epochs = 20
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=num_epochs, lr=1e-3)
    
    save_path = "autonomous_driver.pth"
    torch.save(trained_model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    main()
