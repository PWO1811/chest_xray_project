import json
import os
from datetime import datetime

import pandas as pd

BASE_DIR = r'D:\дз\практика\3 курс\chest_xray_project'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

df = pd.read_csv(os.path.join(RESULTS_DIR, 'results.csv'))

json_data = {
    'timestamp': datetime.now().isoformat(),
    'best_model': df.iloc[0]['Model'],
    'best_accuracy': float(df.iloc[0]['Accuracy']),
    'results': df.to_dict(orient='records')
}

with open(os.path.join(RESULTS_DIR, 'results.json'), 'w', encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

print(f"✅ JSON сохранён: {os.path.join(RESULTS_DIR, 'results.json')}")

with open(os.path.join(RESULTS_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("ОТЧЁТ ПО РЕЗУЛЬТАТАМ ОБУЧЕНИЯ\n")
    f.write("=" * 60 + "\n")
    f.write(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
    f.write(f"Задача: Классификация рентгеновских снимков (норма/пневмония)\n")
    f.write(f"Датасет: PneumoniaMNIST\n")
    f.write(f"Количество классов: 2\n")
    f.write(f"Размер выборки: Train={4708}, Val={524}, Test={624}\n")
    f.write(f"Размер входных изображений: 64x64\n")
    f.write(f"Количество эпох: 5\n")
    f.write(f"Оптимизатор: Adam, lr=0.001\n\n")
    f.write("-" * 60 + "\n")
    f.write("РЕЗУЛЬТАТЫ МОДЕЛЕЙ (сортировка по Accuracy)\n")
    f.write("-" * 60 + "\n")

    for idx, row in df.iterrows():
        f.write(f"{row['Model']}:\n")
        f.write(f"  Accuracy: {row['Accuracy']:.4f} ({row['Accuracy'] * 100:.2f}%)\n")
        f.write(f"  F1-Score: {row['F1_Score']:.4f}\n")
        f.write(f"  Время обучения: {row['Training_Time']:.2f} сек\n\n")

    f.write("-" * 60 + "\n")
    f.write(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {df.iloc[0]['Model']} (Accuracy: {df.iloc[0]['Accuracy'] * 100:.2f}%)\n")
    f.write("-" * 60 + "\n")
    f.write("Ограничения:\n")
    f.write("  - Модель не является диагностическим медицинским инструментом\n")
    f.write("  - Результаты требуют подтверждения врачом-рентгенологом\n")
    f.write("  - Модель обучена на датасете PneumoniaMNIST (28x28 → 64x64)\n")
    f.write("  - Для реальных задач требуется дообучение на более крупных данных\n")

print(f"✅ Текстовый отчёт сохранён: {os.path.join(RESULTS_DIR, 'report.txt')}")
print("\n📁 ВСЕ ФАЙЛЫ В РЕЗУЛЬТАТАХ:")
print("  - results.csv — таблица")
print("  - results.json — JSON-версия")
print("  - report.txt — текстовый отчёт")
print("  - comparison.png — графики")
print("  - cm_*.png — матрицы ошибок (5 штук)")
print("  - demo_results.png — демонстрация работы модели")
print("  - demo_stats.txt — статистика демо-модуля")
