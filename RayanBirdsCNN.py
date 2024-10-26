import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import shutil
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np
import zipfile

#######################################################################################################

 # dl
def download_dataset(dataset_id="RayanAi/Noisy_birds", local_dataset_dir="/content/drive/MyDrive/Noisy_birds"):
    os.makedirs(local_dataset_dir, exist_ok=True)
    with open(os.devnull, 'w') as fnull:
        original_stdout = sys.stdout
        try:
            sys.stdout = fnull
            snapshot_download(repo_id=dataset_id, local_dir=local_dataset_dir, repo_type="dataset")
        finally:
            sys.stdout = original_stdout
    print("Dataset downloaded completely.")

#######################################################################################################

#  training / validation 
def create_and_split_dataset(root_dir="/content/drive/MyDrive/Noisy_birds", split_ratio=0.8):
    classes = ["Budgie", "Rubber Duck", "Canary", "Duckling"]
    labeled_path = os.path.join(root_dir, "labeled")
    train_path = os.path.join(root_dir, "train")
    val_path = os.path.join(root_dir, "val")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    for class_name in classes:
        class_dir = os.path.join(labeled_path, class_name)
        train_class_dir = os.path.join(train_path, class_name)
        val_class_dir = os.path.join(val_path, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Check if the class directory exists to avoid FileNotFoundError
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping this class.")
            continue

        images = [f for f in os.listdir(class_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        train_images, val_images = train_test_split(images, train_size=split_ratio, random_state=42)

        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_class_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_class_dir, img))

    print("Dataset has been split into training and validation sets.")


  # CNN
class Model(nn.Module):
    def __init__(self, num_classes=4):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#  Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# training
def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 30, patience: int = 5, lr: float = 1e-4):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs)

    scores = [[], []]
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * len(inputs)

        avg_train_loss = running_loss / len(train_loader.dataset)
        scores[0].append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * len(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        scores[1].append(avg_val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

 #       if avg_val_loss < best_val_loss:
 #           best_val_loss = avg_val_loss
#          best_model_state = model.state_dict()
 #           counter = 0
 #       else:
  #          counter += 1
  #          if counter >= patience:
  #              print("Early stopping due to no improvement.")
   #             break

    if best_model_state:
        model.load_state_dict(best_model_state)

    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'F1 Score on Validation set: {f1:.4f}')
    accuracy_counter = sum(p == l for p, l in zip(all_preds, all_labels))
    print(f'Accuracy on Validation set: {accuracy_counter / len(all_labels):.4f}')

    return model, scores


def plot_images_with_predictions(images, labels, predicted_labels, classes):
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))  # Create a 5x5 grid
    for i, ax in enumerate(axes.flat):
        if i >= len(images): break
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_title(f"True: {classes[labels[i]]}\nPred: {classes[predicted_labels[i]]}")
        ax.axis("off")
    plt.tight_layout()  #some space 
    plt.show()

if __name__ == "__main__":
    download_dataset()  
    create_and_split_dataset() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # avoiding data leakage
    train_dataset = ImageFolder(root="/content/drive/MyDrive/Noisy_birds/train", transform=transform)
    val_dataset = ImageFolder(root="/content/drive/MyDrive/Noisy_birds/val", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Model(num_classes=4)  # Initialize the model
    trained_model, scores = train(model, train_loader, val_loader, num_epochs=500, patience=5, lr=1e-4)

    # visualize
    classes = train_dataset.classes
    images, labels = next(iter(val_loader))
    images, labels = images[:25], labels[:25]

    trained_model.eval()
    with torch.no_grad():
        outputs = trained_model(images.to(device))
        predicted_labels = outputs.argmax(dim=1)

   
    plot_images_with_predictions(images, labels, predicted_labels.cpu(), classes)

    # saving the model and script
 #   with zipfile.ZipFile('submission.zip', 'w') as zipf:
 #       zipf.write('model.pth')
 #      
  #      zipf.write('model.py')  