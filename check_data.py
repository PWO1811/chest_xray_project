import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST

print("Загрузка данных...")
train_dataset = PneumoniaMNIST(split='train', download=True)
val_dataset = PneumoniaMNIST(split='val', download=True)
test_dataset = PneumoniaMNIST(split='test', download=True)

print(f"Размер обучающей выборки: {len(train_dataset)}")
print(f"Размер валидационной выборки: {len(val_dataset)}")
print(f"Размер тестовой выборки: {len(test_dataset)}")
print(f"Количество классов: {len(train_dataset.info['label'])}")

image, label = train_dataset[0]
print(f"Метка (0 - норма, 1 - пневмония): {label[0]}")

plt.imshow(image, cmap='gray')
plt.title(f"Класс: {'Норма' if label[0] == 0 else 'Пневмония'}")
plt.axis('off')
plt.show()
