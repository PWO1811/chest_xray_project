import os
import time

import matplotlib.pyplot as plt
import pandas as pd
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

BASE_DIR = r'D:\дз\практика\3 курс\chest_xray_project'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001


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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


def replace_last_layer(model, num_classes=2):
    last_layer_name = list(model._modules.keys())[-1]
    last_layer = getattr(model, last_layer_name)

    in_features = None

    if hasattr(last_layer, 'in_features'):
        in_features = last_layer.in_features
    elif isinstance(last_layer, nn.Sequential):
        for sublayer in reversed(last_layer):
            if hasattr(sublayer, 'in_features'):
                in_features = sublayer.in_features
                break
    elif hasattr(last_layer, 'in_channels'):
        in_features = last_layer.in_channels

    if in_features is None:
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            for layer in reversed(model.classifier):
                if hasattr(layer, 'in_features'):
                    in_features = layer.in_features
                    last_layer_name = 'classifier'
                    break
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            last_layer_name = 'fc'

    if in_features is None:
        raise AttributeError(f"Не могу определить входной размер для {last_layer_name}")

    new_layer = nn.Linear(in_features, num_classes)
    setattr(model, last_layer_name, new_layer)
    print(f"  Заменён {last_layer_name} (in={in_features} -> {num_classes} классов)")

    return model


print("\nСоздание моделей...")

model_resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_resnet18 = replace_last_layer(model_resnet18, num_classes=2)

model_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model_resnet50 = replace_last_layer(model_resnet50, num_classes=2)

model_densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
model_densenet = replace_last_layer(model_densenet, num_classes=2)

model_efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model_efficientnet = replace_last_layer(model_efficientnet, num_classes=2)

print("  Создаём MobileNetV3...")
model_mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

in_features_mobilenet = 960
model_mobilenet.classifier = nn.Sequential(
    nn.Linear(in_features_mobilenet, 1280),
    nn.Hardswish(),
    nn.Dropout(p=0.2),
    nn.Linear(1280, 2)
)
print(f"  Заменён classifier (in=960 -> 2 классов)")

models_dict = {
    'ResNet18': model_resnet18,
    'ResNet50': model_resnet50,
    'DenseNet121': model_densenet,
    'EfficientNet_B0': model_efficientnet,
    'MobileNetV3': model_mobilenet
}


def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
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

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return model, history


def test_model(model, test_loader):
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

    return accuracy, f1, cm


results = {
    'Model': [],
    'Accuracy': [],
    'F1_Score': [],
    'Training_Time': []
}

for name, model in models_dict.items():
    print(f"\n{'=' * 50}")
    print(f"Обучаем модель: {name}")
    print(f"{'=' * 50}")

    start_time = time.time()

    trained_model, history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE
    )

    accuracy, f1, cm = test_model(trained_model, test_loader)

    end_time = time.time()
    training_time = end_time - start_time

    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['F1_Score'].append(f1)
    results['Training_Time'].append(training_time)

    print(f"✓ {name} | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Time: {training_time:.2f}s")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'cm_{name}.png'), dpi=150)
    plt.close()

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(RESULTS_DIR, 'results.csv'), index=False)

df_sorted = df_results.sort_values('Accuracy', ascending=False)
print("\n" + "=" * 50)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 50)
print(df_sorted.to_string(index=False))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
bars = plt.bar(df_results['Model'], df_results['Accuracy'], color='skyblue')
plt.ylim(0.7, 0.95)
plt.title('Сравнение точности (Accuracy)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
for bar, acc in zip(bars, df_results['Accuracy']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

plt.subplot(1, 2, 2)
bars = plt.bar(df_results['Model'], df_results['F1_Score'], color='lightgreen')
plt.ylim(0.7, 0.95)
plt.title('Сравнение F1-Score')
plt.ylabel('F1-Score')
plt.xticks(rotation=45)
for bar, f1 in zip(bars, df_results['F1_Score']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{f1:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparison.png'), dpi=150)
plt.show()

best_model = df_sorted.iloc[0]['Model']
best_acc = df_sorted.iloc[0]['Accuracy']

print(f"\n{'=' * 50}")
print("✅ ВСЕ МОДЕЛИ ОБУЧЕНЫ!")
print(f"🏆 Лучшая модель: {best_model} (Accuracy: {best_acc:.4f})")
print(f"📁 Результаты в: {RESULTS_DIR}")
print("   - results.csv")
print("   - comparison.png")
print("   - cm_*.png (5 матриц ошибок)")
print("=" * 50)
