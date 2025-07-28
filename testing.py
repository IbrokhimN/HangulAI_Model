import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

MODEL_PATH = "model.pth"
TEST_PATH = "/home/ibrokhim/Документы/test"
WRONG_PATH = "wrong"
BATCH_SIZE = 64

if os.path.exists(WRONG_PATH):
    shutil.rmtree(WRONG_PATH)
os.makedirs(WRONG_PATH, exist_ok=True)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
test_dataset = datasets.ImageFolder(TEST_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
class_names = test_dataset.classes
num_classes = len(class_names)

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
correct = 0
total = 0
image_index = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(images.size(0)):
            total += 1
            if predicted[i] == labels[i]:
                correct += 1
            else:
                actual_class = class_names[labels[i]]
                predicted_class = class_names[predicted[i]]
                wrong_dir = os.path.join(WRONG_PATH, f"real_{actual_class}_pred_{predicted_class}")
                os.makedirs(wrong_dir, exist_ok=True)
                save_image(images[i], os.path.join(wrong_dir, f"{image_index}.png"))
            image_index += 1

accuracy = correct / total * 100
print(f"the test is over : accuracy {accuracy:.2f}% ({correct}/{total})")
print(f"wrong ones were saved at '{WRONG_PATH}'")

