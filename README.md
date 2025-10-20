## Цель проекта

Создать модель машинного обучения, которая по тексту песни определяет её музыкальный жанр
Создать удобный интерфейс для работы с моделью

## Данные

Для обучения был взят [датасет с kaggle], который содержит более 3 миллионов текстов песен на английском и русском языках
После EDA я изменил этот датасет и опубликовал [новый] на hf
Он содержит 43 разных жанров, причем одна песня может принадлежать нескольким жанрам

## Целевые метрики

Среднее время отклика сервиса ≤ 200 мс
Доля неуспешных запросов ≤ 1 %
Использование памяти/CPU — в пределах SLA
Качество модели: accuracy ≥ 90 %, micro-f1 ≥ 80 %

## Запуск обучения

Для запуска обучения необходимо склонировать репозиторий, установить зависимости, залогиниться в hf (если хотите опубликовать модель на hf) и перейти в директорию train

```sh
git clone https://github.com/Yegor5/lyrics_genre_classification.git
cd lyrics_genre_classification
pip install -r requirements.txt
huggingface-cli login
cd train
```

Далее запустить обучение можно командой 

```sh
python train.py --config config.yaml --verbose
```

Параметры обучения и пути для сохранения модели, логирования обучения и загрузки датасета лежат в файле config.yaml

[датасет с kaggle]: <https://www.kaggle.com/datasets/travissscottt/ru-and-en-song-lyrics-for-genre-classification?resource=download>
[новый]: <https://huggingface.co/datasets/Yegor25/lyrics_genre_dataset>
