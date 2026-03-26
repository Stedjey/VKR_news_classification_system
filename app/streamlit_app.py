import os
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "rubert_final_model"
LABEL_ENCODER_PATH = BASE_DIR / "rubert_label_encoder.pkl"
DB_DIR = Path(os.getenv("APP_DB_DIR", str(BASE_DIR / "db")))
if not DB_DIR.is_absolute():
    DB_DIR = BASE_DIR / DB_DIR
DB_DIR.mkdir(parents=True, exist_ok=True)
STREAMLIT_DB_PATH = DB_DIR / "streamlit_messages.db"

# Загрузка модели и энкодера
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Загрузка модели для анализа настроений
sentiment_model = "sismetanin/rubert-ru-sentiment-rusentiment"
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)

# Загрузка модели для анализа эмоций
emotion_model = pipeline(model="seara/rubert-tiny2-ru-go-emotions")
emotion_labels_map = {
    'admiration': 'восхищение', 'amusement': 'веселье', 'anger': 'злость', 'annoyance': 'раздражение',
    'approval': 'одобрение', 'caring': 'забота', 'confusion': 'непонимание', 'curiosity': 'любопытство',
    'desire': 'желание', 'disappointment': 'разочарование', 'disapproval': 'неодобрение', 'disgust': 'отвращение',
    'embarrassment': 'смущение', 'excitement': 'возбуждение', 'fear': 'страх', 'gratitude': 'признательность',
    'grief': 'горе', 'joy': 'радость', 'love': 'любовь', 'nervousness': 'нервозность', 'optimism': 'оптимизм',
    'pride': 'гордость', 'realization': 'осознание', 'relief': 'облегчение', 'remorse': 'раскаяние',
    'sadness': 'грусть', 'surprise': 'удивление', 'neutral': 'нейтральность'
}

# Функция классификации текста
def classify_text(text: str):
    # Обрезаем входной текст до 512 токенов
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()  # Перемещаем тензор на CPU, если использован GPU
    probs = softmax(logits, axis=1)
    # Индекс класса с максимальной вероятностью
    pred_idx = np.argmax(probs)
    # Название класса
    class_name = le.inverse_transform([pred_idx])[0]
    # Уверенность модели
    confidence = probs[0][pred_idx]
    return class_name, confidence

# Функция для анализа настроений
def get_sentiment(text):
    sentiment = sentiment_analyzer(text, truncation=True, max_length=512)
    labels = {'LABEL_0': 'Отрицательное', 'LABEL_1': 'Нейтральное', 'LABEL_2': 'Положительное'}
    label = sentiment[0]['label']
    confidence = sentiment[0]['score']
    return labels[label], confidence

# Функция для анализа эмоций
def get_emotion(text):
    result = emotion_model(text, truncation=True, max_length=512)
    translated_label = emotion_labels_map.get(result[0]['label'], result[0]['label'])
    return translated_label, result[0]['score']

# Функция для получения важных слов
def get_word_importance(text):
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, padding=True, max_length=256)

    embedding_layer = model.get_input_embeddings()
    embedded = embedding_layer(inputs['input_ids'])
    embedded.requires_grad_()
    embedded.retain_grad()

    outputs = model(inputs_embeds=embedded, attention_mask=inputs['attention_mask'])
    pred_class = torch.argmax(outputs.logits, dim=1)
    outputs.logits[0, pred_class].backward()

    grads = embedded.grad[0]
    grads_norm = grads.norm(dim=1)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    offsets = inputs['offset_mapping'][0].tolist()

    word_scores = {}
    current_word = ""
    current_score = 0
    current_len = 0

    for token, offset, score in zip(tokens, offsets, grads_norm):
        if token in ["[CLS]", "[SEP]"]:
            continue
        if token.startswith("##"):
            current_word += token[2:]
            current_score += score.item()
            current_len += 1
        else:
            if current_word:
                word_scores[current_word] = current_score / max(current_len, 1)
            current_word = token
            current_score = score.item()
            current_len = 1

    if current_word:
        word_scores[current_word] = current_score / max(current_len, 1)

    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    return sorted_words[:5]

# Функция для создания таблицы в БД (если она еще не существует)
def create_table():
    try:
        conn = sqlite3.connect(STREAMLIT_DB_PATH)
        cursor = conn.cursor()

        # Запрос для создания таблицы
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS streamlit_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                predicted_category TEXT,
                confidence REAL,
                keywords TEXT,
                emotion TEXT,
                sentiment TEXT
            )
        ''')

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Ошибка при создании таблицы в БД: {e}")

# Вызов функции для создания таблицы
create_table()

# Функция сохранения в БД для Streamlit
def save_to_db(text, category, confidence, keywords, emotion, sentiment):
    try:
        # Подключаемся к базе данных
        conn = sqlite3.connect(STREAMLIT_DB_PATH)
        cursor = conn.cursor()
        
        # Выполняем SQL запрос на вставку данных в таблицу
        cursor.execute('''
            INSERT INTO streamlit_messages (text, predicted_category, confidence, keywords, emotion, sentiment)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            text,
            category,
            round(float(confidence), 2),
            keywords,
            emotion,
            sentiment
        ))
        
        # Сохраняем изменения
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Ошибка при сохранении данных в БД: {e}")

# Streamlit интерфейс
st.title("Классификатор текстов, анализ тональности и эмоций")

# Ввод текста
input_text = st.text_area("Введите текст для анализа:", height=200)

if st.button("Предсказать"):
    if input_text:
        # Классификация текста
        class_name, confidence = classify_text(input_text)

        # Эмоция текста
        emotion, emotion_confidence = get_emotion(input_text)

        # Настроение текста
        sentiment, sentiment_confidence = get_sentiment(input_text)

        # Важные слова по мнению модели
        important_words = get_word_importance(input_text)
        words_str = ', '.join([word for word, _ in important_words])


        # Формирование ответа
        response = (
            f"🧠 Предсказанная категория: *{class_name}*\n"
            f"📊 Уверенность: *{confidence:.2%}*\n\n"
            f"🔍 Важные слова по мнению модели:\n"
            f"{words_str}\n\n"
            f"💭 Тональность текста: *{sentiment}* с уверенностью {sentiment_confidence:.2f}\n"
            f"😌 Эмоция текста: *{emotion}* с уверенностью {emotion_confidence:.2f}\n"
        )

        st.markdown(response)

        # Сохраняем данные в БД
        save_to_db(
            text=input_text,
            category=class_name,
            confidence=confidence,
            keywords=words_str,  # Передаем ключевые слова
            emotion=emotion,  # Сохраняем эмоцию
            sentiment=sentiment  # Сохраняем настроение
        )

    else:
        st.warning("Пожалуйста, введите текст для анализа.")
