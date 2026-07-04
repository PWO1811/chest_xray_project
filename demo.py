import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from medmnist import PneumoniaMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {DEVICE}")

BASE_DIR = r'D:\дз\практика\3 курс\chest_xray_project'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class GrayscaleToRGB:
    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)


test_transform = transforms.Compose([
    transforms.ToTensor(),
    GrayscaleToRGB(),
    transforms.Resize((64, 64)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("Загрузка тестовых данных...")
test_dataset = PneumoniaMNIST(split='test', transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

print(f"Тестовая выборка: {len(test_dataset)} изображений")

print("\nЗагрузка модели ResNet50...")

from torchvision import models

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

weights_path = os.path.join(MODELS_DIR, 'ResNet50.pth')
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print(f"✅ Веса загружены из {weights_path}")
else:
    print(f"❌ Файл с весами не найден: {weights_path}")
    print("   Сначала обучи модель через train_all_models.py")
    exit()

model = model.to(DEVICE)
model.eval()


def predict_image(model, image, true_label=None):
    """Принимает изображение, возвращает предсказание и уверенность"""
    with torch.no_grad():
        image = image.to(DEVICE)
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

        predicted_class = predicted.item()
        confidence_score = confidence.item() * 100

        return predicted_class, confidence_score


print("\n" + "=" * 60)
print("ДЕМОНСТРАЦИОННЫЙ МОДУЛЬ - КЛАССИФИКАЦИЯ РЕНТГЕНОВСКИХ СНИМКОВ")
print("=" * 60)

total_processed = 0
correct_predictions = 0
results_history = []

num_examples = 10

indices = random.sample(range(len(test_dataset)), num_examples)

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    image, label = test_dataset[indices[idx]]
    true_label = label.item()
    predicted_class, confidence = predict_image(model, image.unsqueeze(0))
    total_processed += 1
    if predicted_class == true_label:
        correct_predictions += 1
    results_history.append({
        'image_idx': indices[idx],
        'true_label': true_label,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'correct': predicted_class == true_label
    })
    img_display = image.squeeze().cpu().numpy()
    img_display = img_display[0]

    ax.imshow(img_display, cmap='gray')
    is_correct = predicted_class == true_label
    color = 'green' if is_correct else 'red'
    label_names = ['Норма', 'Пневмония']
    ax.set_title(f"Истинный: {label_names[true_label]}\n"
                 f"Предсказание: {label_names[predicted_class]}\n"
                 f"Уверенность: {confidence:.1f}%",
                 color=color, fontsize=10)
    ax.axis('off')

plt.suptitle(f'ДЕМОНСТРАЦИЯ РАБОТЫ МОДЕЛИ ResNet50\n'
             f'Всего показано: {total_processed}, '
             f'Правильно: {correct_predictions}, '
             f'Точность: {correct_predictions / total_processed * 100:.1f}%',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'demo_results.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("КРАТКАЯ СТАТИСТИКА ПОСЛЕДНЕГО ЗАПУСКА")
print("=" * 60)
print(f"Обработано изображений: {total_processed}")
print(f"Правильных предсказаний: {correct_predictions}")
print(f"Точность на выборке: {correct_predictions / total_processed * 100:.1f}%")
print(f"Ошибок: {total_processed - correct_predictions}")

print("\nДетали по каждому изображению:")
print("-" * 70)
print(f"{'№':<4} {'Истинный':<12} {'Предсказанный':<15} {'Уверенность':<12} {'Результат'}")
print("-" * 70)

for i, res in enumerate(results_history):
    label_names = ['Норма', 'Пневмония']
    status = '✅ Правильно' if res['correct'] else '❌ Ошибка'
    print(f"{i + 1:<4} {label_names[res['true_label']]:<12} "
          f"{label_names[res['predicted_class']]:<15} "
          f"{res['confidence']:.1f}%{' ':<8} {status}")

print("-" * 70)

with open(os.path.join(RESULTS_DIR, 'demo_stats.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("ДЕМОНСТРАЦИОННЫЙ МОДУЛЬ - СТАТИСТИКА\n")
    f.write("=" * 60 + "\n")
    f.write(f"Модель: ResNet50\n")
    f.write(f"Обработано изображений: {total_processed}\n")
    f.write(f"Правильных предсказаний: {correct_predictions}\n")
    f.write(f"Точность: {correct_predictions / total_processed * 100:.1f}%\n")
    f.write(f"Ошибок: {total_processed - correct_predictions}\n\n")
    f.write("Детали:\n")
    for i, res in enumerate(results_history):
        label_names = ['Норма', 'Пневмония']
        status = 'Правильно' if res['correct'] else 'ОШИБКА'
        f.write(f"{i + 1}. Истинный: {label_names[res['true_label']]}, "
                f"Предсказанный: {label_names[res['predicted_class']]}, "
                f"Уверенность: {res['confidence']:.1f}%, {status}\n")

print(f"\n✅ Статистика сохранена в: {os.path.join(RESULTS_DIR, 'demo_stats.txt')}")
print(f"✅ Изображение с результатами сохранено в: {os.path.join(RESULTS_DIR, 'demo_results.png')}")
print("\n🎯 ДЕМОНСТРАЦИОННЫЙ МОДУЛЬ ЗАВЕРШЁН!")
