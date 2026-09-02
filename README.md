# LLM  
**Коллекция и реализация Large Language Models с нуля**

Этот репозиторий предназначен для реализации, обучения и экспериментов с различными архитектурами **Large Language Models (LLM)**.  
Цель проекта — **глубокое понимание внутреннего устройства LLM**, а не просто использование готовых библиотек.

Каждая модель реализуется:
- в отдельной папке,
- с собственной архитектурой,
- своим обучающим кодом,
- своим токенизатором и чекпоинтами.

Репозиторий масштабируется: новые модели будут добавляться постепенно (GPT-2, LLaMA, Mistral, Mixtral, Gemma и др.).

---

## 📁 Общая структура репозитория
```text
llm/
├── README.md
├── requirements.txt                                    
└── models/
    ├── gpt1/                  # Реализация GPT-1
    │   ├── checkpoints/       # Сохранённые веса моделей
    │   │   └── gpt_checkpoint.pt
    │   └── src/
    │       ├── bpe/           # Byte Pair Encoding токенизатор
    │       │   ├── __init__.py
    │       │   └── bpe.py
    │       ├── model/         # Архитектура модели
    │       │   ├── __init__.py
    │       │   └── gpt.py
    │       └── scripts/       # Обучение и генерация
    │           ├── __init__.py
    │           ├── train.py
    │           └── generate.py
    │
    ├── gpt2/                  
    ├── llama/                 # (планируется)
    ├── mistral/               # (планируется)
    ├── mixtral/               # (планируется)
    └── gemma/                 # (планируется)
```


Каждая модель полностью **изолирована** и не зависит от других.

---

# Реализованные модели

---

<details>
<summary><strong> GPT-1 </strong></summary>

### Архитектура

- Decoder-only Transformer
- Masked Multi-Head Self-Attention
- Residual connections
- LayerNorm
- FeedForward блоки
- Классическая GPT-логика (next-token prediction)

---

### 📂 Структура `models/gpt1`
**Датасет:** [Russian Novels](https://github.com/JoannaBy/RussianNovels/tree/master)
```text
models/gpt1/
├── checkpoints/
│ └── gpt_checkpoint.pt # веса модели
└── src/
├── bpe/
│ ├── init.py
│ └── bpe.py # реализация Byte Pair Encoding
├── model/
│ ├── init.py
│ └── gpt.py # модель GPT + Dataset + train/generate
└── scripts/
├── init.py
├── train.py # обучение модели
└── generate.py # генерация текста
```

---

### 🧾 Данные

- `tokenizer.json`  
  Содержит:
  - `token2id`
  - `id2token`
  - `vocab_size`

- `token_ids.pt`  
  Один длинный тензор токенов всего корпуса  
  Используется для формирования train / validation выборок

---

### 🚀 Обучение

Запуск обучения из корня репозитория:

```bash
python -m models.gpt1.src.scripts.train
```
✨ Генерация текста
```bash
python -m models.gpt1.src.scripts.generate
```
⚙️ Пример параметров обучения

| Параметр      | Значение      |
| ------------- | ------------- |
| vocab_size    | 2000–3000     |
| seq_len       | 64–256        |
| emb_size      | 256–512       |
| num_heads     | 4–8           |
| num_layers    | 4–12          |
| dropout       | 0.1–0.2       |
| learning_rate | 1e-5 – 2.5e-4 |
| batch_size    | 32–128        |

</details>

<details>
<summary><strong> GPT-2 </strong></summary>

### Архитектура

- Decoder-only Transformer
- Masked Multi-Head Self-Attention
- KV-cache (Key / Value caching)
- Residual connections
- LayerNorm (Pre-LN)
- FeedForward блоки (Linear → GELU → Linear)
- next-token prediction
---

### 📂 Структура `models/gpt2`
**Датасет:** [Russian Novels](https://github.com/JoannaBy/RussianNovels/tree/master)
```text
models/gpt2/
├── checkpoints/
│   └── gpt2_checkpoint.pt        # веса модели
├── data/
│   ├── corpus/                   # исходные тексты 
├── src/
│   ├── bpe/
│   │   ├── bpe.py                # реализация BPE
│   │   └── tokenizer_generate.py # обучение токенайзера
│   ├── model/
│   │   ├── __init__.py
│   │   ├── activations.py        # GELU
│   │   └── gpt2.py               # модель GPT-2 + KV-cache
│   └── scripts/
│       ├── encode_corpus.py      # кодирование корпуса в token_ids.pt
│       ├── train.py              # обучение модели
│       └── generate.py           # генерация текста
```

---

### 🧾 Данные

- `tokenizer.json`  
  Содержит:
  - `token2id`
  - `id2token`
  - `vocab_size`

- `token_ids.pt`  
  Один длинный тензор токенов всего корпуса  
  Используется для формирования train / validation выборок

---

### 🚀 Обучение

Запуск обучения из корня репозитория:

```bash
python -m models.gpt2.src.scripts.train
```
✨ Генерация текста
```bash
python -m models.gpt2.src.scripts.generate
```
⚙️ Пример параметров обучения

| Параметр      | Значение          |
| ------------- | ----------------- |
| vocab_size    | из tokenizer.json |
| seq_len       | 128 – 256         |
| emb_size      | 256               |
| num_heads     | 4 – 8             |
| num_layers    | 4 – 8             |
| dropout       | 0.1               |
| learning_rate | 3e-4              |
| batch_size    | 32 – 64           |

</details>
