import torch
import torchvision
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import warnings
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

from config import get_config, get_weights_file_path
from model import VisionTransformer

from dataset import CIFAR10


def train_model(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # Transformations
    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]
    )

    # Load the dataset
    main_dataset = load_dataset("cifar10")

    # Get training and validation dataset
    train_set, val_set = main_dataset["train"], main_dataset["test"]

    # Create custom datasets
    train_dataset = CIFAR10(train_set, transform=transform)
    val_dataset = CIFAR10(val_set, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Instantiate the model and move to GPU if available
    model = VisionTransformer(config)
    model.to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = GradScaler(enabled=True)  # For automatic mixed precision

    grad_accumulation_steps = config["batch_size"]
    num_epochs = config["num_epoch"]

    training_accuracies = []
    training_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        total_loss, total_accuracy = 0, 0
        main_total = 0
    
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch (train) {epoch:02d}")
        for step, (images, labels) in enumerate(batch_iterator):
            images, labels = images.to(device), labels.to(device)

            # Forward pass with automatic mixed precision
            with autocast(enabled=True):
                output = model(images)
                loss = criterion(output, labels)

            # Backward pass and optimization with gradient accumulation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()
            
            predicted = (output.argmax(dim=1) == labels).float().mean()
            total_accuracy += predicted.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / main_total

        training_accuracies.append(avg_train_accuracy)
        training_losses.append(avg_train_loss)


        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss}, Accuracy: {avg_train_accuracy}")

        model.eval()
        val_loss, val_accuracy = 0, 0
        total_val_accuracy = 0

        with torch.no_grad():
            batch_iterator = tqdm(test_loader, desc=f"Processing epoch (test) {epoch:02d}")
            for step, (images, labels) in enumerate(batch_iterator):

                images, labels = images.to(device), labels.to(device)

                output = model(images)
                loss = criterion(output, labels)

                val_loss += loss.item() # * grad_accumulation_steps

                _, predicted = output.max(1)

                val_accuracy += predicted.eq(labels).sum().item()
                total_val_accuracy += labels.size(0)

        # Calculate average loss and accuracy for the validation
        avg_val_loss = val_loss / len(test_loader)
        avg_val_accuracy = val_accuracy / total_val_accuracy

        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        print(f"Val Loss: {avg_val_loss}, Val Accuracy: {avg_val_accuracy}")

        model_filename = get_weights_file_path(config, f"{epoch}")

        torch.save(
            {
                # "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # "global_Step": global_step,
            },
            model_filename,
        )

        total_loss, total_accuracy, main_total = 0, 0, 0

    # Open a text file in write mode
    # Add performance data in text
    with open("data.txt", "w") as file:
        # Loop through the array and write each element to the file
        res = "training_accuracies = ["
        for data in training_accuracies:
            res += str(data) + ", "
        res = "]\ntraining_losses = ["
        for data in training_losses:
            res += str(data) + ", "
        res += "]\nval_losses = ["
        for data in val_losses:
            res += str(data) + ", "
        res += "]\nval_accuracies = ["
        for data in val_accuracies:
            res += str(data) + ", "
        res += "]\n"
            
        file.write(res)  # Write the item followed by a newline character

    print("Data has been written to data.txt")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
