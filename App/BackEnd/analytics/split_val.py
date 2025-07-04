import os
import random
import shutil
from pathlib import Path

# Настройки
random.seed(42)
train_dir = Path("K:\Diploma_new\datasets\\al5083\weld\\train\sorted_images")
val_dir = Path("K:\Diploma_new\datasets\\al5083\weld\\val\sorted_images")
val_split = 0.2  # 20% от каждого класса

# Создание директории для val
val_dir.mkdir(parents=True, exist_ok=True)

# Перебираем классы
for class_name in os.listdir(train_dir):
    class_train_path = train_dir / class_name
    class_val_path = val_dir / class_name

    if not class_train_path.is_dir():
        continue

    # Создаём папку под класс в val, если не существует
    class_val_path.mkdir(parents=True, exist_ok=True)

    # Получаем список файлов класса
    images = list(class_train_path.glob("*"))
    num_val = int(len(images) * val_split)

    # Случайно выбираем val-изображения
    val_images = random.sample(images, num_val)

    print(f"Класс: {class_name} — перемещается в val: {num_val} изображений")

    # Перемещаем файлы
    for img_path in val_images:
        shutil.move(str(img_path), str(class_val_path / img_path.name))

print("\n✅ Перемещение завершено.")
