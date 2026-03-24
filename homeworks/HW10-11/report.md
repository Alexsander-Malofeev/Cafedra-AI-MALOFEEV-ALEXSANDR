# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Выбран датасет для части A: STL10, так как он рекомендован в задании и хорошо подходит для сравнения простой CNN, аугментаций и transfer learning.
- Выбран датасет для части B: Pascal VOC, трек `segmentation`, так как он естественно сочетается с pretrained segmentation-моделью из `torchvision`.
- В части A сравнивались: простая CNN без аугментаций, та же CNN с аугментациями, ResNet18 с замороженным backbone и ResNet18 с partial fine-tuning.
- Во второй части сравнивались два режима постобработки сегментации: базовая маска и альтернативная морфологическая очистка.

## 2. Среда и воспроизводимость

- Python: (укажите версию Python)
- torch / torchvision: (укажите версии torch и torchvision)
- Устройство (CPU/GPU): (подставьте из вывода ноутбука)
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: STL10
- Разделение: train/val/test = 80/20 split от train + официальный test
- Базовые transforms: `Resize((96, 96)) + ToTensor + Normalize`
- Augmentation transforms: `Resize((96, 96)) + RandomHorizontalFlip + RandomCrop + ColorJitter + ToTensor + Normalize`
- Комментарий (2-4 предложения): STL10 содержит 10 классов цветных изображений размером 96x96. Датасет относительно небольшой, поэтому хорошо видно влияние аугментаций и transfer learning. Для простой CNN задача решаемая, но pretrained ResNet18 обычно даёт более сильный baseline.

### 3.2. Часть B: structured vision

- Датасет: Pascal VOC
- Трек: segmentation
- Что считается ground truth: бинарная маска foreground-класса `person` по разметке VOCSegmentation
- Какие предсказания использовались: бинарная маска `person`, полученная из `argmax` по выходу pretrained DeepLabV3_ResNet50
- Комментарий (2-4 предложения): постановка с одним foreground-классом `person` удобна для базовой оценки сегментации. Модель `DeepLabV3_ResNet50` из `torchvision` обучена на VOC-совместимых классах, поэтому такая проверка корректна и интерпретируема.

## 4. Часть A: модели и обучение (C1-C4)

Опишите коротко и сопоставимо:

- C1 (simple-cnn-base): простая CNN без аугментаций
- C2 (simple-cnn-aug): та же CNN + аугментации
- C3 (resnet18-head-only): pretrained ResNet18, backbone заморожен, обучается только `fc`
- C4 (resnet18-finetune): pretrained ResNet18, обучаются `layer4 + fc`

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer(ы): Adam
- Batch size: (подставьте из ноутбука)
- Epochs (макс): (подставьте из ноутбука)
- Критерий выбора лучшей модели: best_val_accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран detection track

- Модель: не использовалось
- V1: `score_threshold = 0.3` — не использовалось
- V2: `score_threshold = 0.7` — не использовалось
- Как считался IoU: не использовалось
- Как считались precision / recall: не использовалось

### Если выбран segmentation track

- Модель: DeepLabV3_ResNet50 (pretrained)
- Что считается foreground: класс `person`
- V1: базовая постобработка, маска `argmax == person`
- V2: альтернативная постобработка, морфологическая очистка бинарной маски
- Как считался mean IoU: IoU считался между предсказанной и истинной бинарной маской `person`, затем усреднялся по изображениям
- Считались ли дополнительные pixel-level метрики: да, pixel precision и pixel recall

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/segmentation_examples.png`, `./artifacts/figures/segmentation_metrics.png`

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: ________
- Лучшая `val_accuracy`: ________
- Итоговая `test_accuracy` лучшего классификатора: ________
- Что дали аугментации (C2 vs C1): ________
- Что дал transfer learning (C3/C4 vs C1/C2): ________
- Что оказалось лучше: head-only или partial fine-tuning: ________
- Что показал режим V1 во второй части: ________
- Что показал режим V2 во второй части: ________
- Как интерпретируются метрики второй части: ________

## 7. Анализ

(8-15 предложений)

Нужно прокомментировать:

- почему простая CNN ведёт себя именно так на выбранном датасете;
- дали ли аугментации устойчивое улучшение;
- почему pretrained ResNet18 помогла или не помогла;
- чем head-only отличается от partial fine-tuning в ваших результатах;
- почему выбранная метрика второй части подходит под задачу;
- что произошло при переходе от V1 к V2;
- какие ошибки модели оказались наиболее показательными.

## 8. Итоговый вывод

(3-7 предложений)

- Какой конфиг классификации вы бы взяли как базовый и почему.
- Что главное вы поняли про transfer learning.
- Что главное вы поняли про detection/segmentation и метрики для этих задач.

## 9. Приложение (опционально)

Если вы делали дополнительные сравнения:

- дополнительные fine-tuning сценарии
- confusion matrix для классификации
- дополнительная постобработка для второй части
- дополнительные графики: `./artifacts/figures/...`