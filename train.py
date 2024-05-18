import torch
import torchvision
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    scaler = GradScaler()  # For automatic mixed precision

    grad_accumulation_steps = config["batch_size"]
    num_epochs = config["num_epoch"]

    # Training loop
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0

        counter = 0
        batch_iterator = tqdm(train_loader, desc=f"Processing epoch {epoch:02d}")
        for step, (images, labels) in enumerate(batch_iterator):
            images, labels = images.to(device), labels.to(device)

            # Forward pass with automatic mixed precision
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass and optimization with gradient accumulation
            scaler.scale(loss).backward()
            if (step + 1) % grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accumulation_steps

            counter += 1

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                val_loss += criterion(output, labels).item()
                val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()

        # Calculate average loss and accuracy for the validation
        val_loss /= len(test_loader)
        val_accuracy /= len(test_loader)

        print(f"Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

        model_filename = get_weights_file_path(config, f"{counter}")

        torch.save(
            {
                # "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # "global_Step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
