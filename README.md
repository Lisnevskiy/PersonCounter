# PersonCounter

## Описание

Проект PersonCounter предназначен для подсчета количества людей, входящих и выходящих из помещения, на основе данных, предоставленных системой компьютерного зрения.

Алгоритм анализирует JSON-файлы с детекциями объектов, пересекающих заданные линии (входа и выхода), и выдает итоговые метрики: количество вошедших, вышедших и текущих посетителей в помещении.


## 🛠 Установка и настройка

### 1. Клонирование репозитория

```bash
git clone https://github.com/Lisnevskiy/PersonCounter.git
```

### 2. Установка зависимостей

Убедитесь, что у вас установлен [Pipenv](https://pipenv.pypa.io/en/latest/). Установите зависимости:

```bash
pipenv install
```

## 🚀 Запуск скрипта:

```bash
python main.py
```

## 🔄 Запуск тестов:

```bash
pytest test_main.py
```