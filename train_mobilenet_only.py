import time

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from medmnist import PneumoniaMNIST
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms, models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {DEVICE}")


class GrayscaleToRGB:
    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)


train_transform = transforms.Compose([
    transforms.ToTensor(),
    GrayscaleToRGB(),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    GrayscaleToRGB(),
    transforms.Resize((64, 64)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Загрузка данных...")
train_dataset = PneumoniaMNIST(split='train', transform=train_transform, download=True)
val_dataset = PneumoniaMNIST(split='val', transform=test_transform, download=True)
test_dataset = PneumoniaMNIST(split='test', transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

print("\nСоздание MobileNetV3...")
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

in_features = 960
print(f"Входной размер классификатора: {in_features}")

model.classifier = nn.Sequential(
    nn.Linear(in_features, 1280),
    nn.Hardswish(),
    nn.Dropout(p=0.2),
    nn.Linear(1280, 2)
)

model = model.to(DEVICE)
print("✓ MobileNetV3 - подготовлена")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n" + "=" * 50)
print("Обучаем модель: MobileNetV3")
print("=" * 50)

start_time = time.time()

for epoch in range(5):
    model.train()
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = train_correct / train_total
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    print(f"Epoch {epoch + 1}/5 | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
cm = confusion_matrix(all_labels, all_preds)

end_time = time.time()
training_time = end_time - start_time

print(f"\n✅ MobileNetV3 | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Time: {training_time:.2f}s")
print(f"Confusion Matrix:\n{cm}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.title('Confusion Matrix - MobileNetV3')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(r'D:\дз\практика\3 курс\chest_xray_project\results\cm_MobileNetV3.png', dpi=150)
plt.close()

print("Матрица ошибок сохранена в results/cm_MobileNetV3.png")
