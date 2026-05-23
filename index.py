from flask import Flask, request, jsonify, render_template
import os
import re
import json
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Корректное отображение кириллицы/символов в JSON

print("Инициализация сервиса и загрузка стеммера...")
stemmer = SnowballStemmer("english")

def clean_and_stem(text):
    if not text or pd.isna(text):
        return ""
    # Очищаем текст от знаков препинания и переводим в нижний регистр
    text = re.sub(r'[^\w\s]', ' ', str(text).lower())
    # Приводим слова к корню
    tokens = [stemmer.stem(word) for word in text.split() if word.strip()]
    return " ".join(tokens)

# Загружаем датасет
csv_path = 'tmdb_5000_movies.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"Успешно загружено фильмов: {len(df)}")
except FileNotFoundError:
    print(f"Критическая ошибка: Файл {csv_path} не найден.")
    raise

# Заполняем пустые значения
df['title'] = df['title'].fillna('')
df['overview'] = df['overview'].fillna('')

print("Предобработка текста (очистка и стемминг)...")
df['norm_title'] = df['title'].apply(clean_and_stem)
df['norm_overview'] = df['overview'].apply(clean_and_stem)

print("Векторизация баз данных TF-IDF...")
# Векторизатор для описаний (overview) — учим биграммы, убираем стоп-слова
tf_overview = TfidfVectorizer(
    analyzer='word', stop_words='english', min_df=1,
    ngram_range=(1, 2), token_pattern=u'(?ui)\\b\\w*[a-z0-9]'
)
tfidf_overview_matrix = tf_overview.fit_transform(df['norm_overview'])

# Отдельный векторизатор для названий (title)
tf_title = TfidfVectorizer(
    analyzer='word', stop_words='english', min_df=1,
    ngram_range=(1, 2), token_pattern=u'(?ui)\\b\\w*[a-z0-9]'
)
tfidf_title_matrix = tf_title.fit_transform(df['norm_title'])

print(f"Матрица описаний: {tfidf_overview_matrix.shape} | Матрица названий: {tfidf_title_matrix.shape}")


def get_search_and_recommendations(user_query, top_n=10):
    query_norm = clean_and_stem(user_query)
    if not query_norm.strip():
        return None

    # ---- ШАГ 1: ПОЛНОСТЬЮ СКВОЗНОЙ ГИБРИДНЫЙ ПОИСК ЦЕЛЕВОГО ФИЛЬМА ----
    user_query_clean = user_query.strip().lower()

    # 1. Считаем семантическое сходство через TF-IDF матрицы
    query_title_vector = tf_title.transform([query_norm])
    query_overview_vector = tf_overview.transform([query_norm])

    sim_title = linear_kernel(query_title_vector, tfidf_title_matrix).flatten()
    sim_overview = linear_kernel(query_overview_vector, tfidf_overview_matrix).flatten()

    # 2. Добавляем строковый поиск (бонусные баллы за вхождение подстроки в реальный текст)
    # Это защищает от «шума» TF-IDF при коротких запросах
    contains_title_bonus = df['title'].str.lower().str.contains(user_query_clean, na=False).astype(float)
    contains_overview_bonus = df['overview'].str.lower().str.contains(user_query_clean, na=False).astype(float)

    # Бонус за абсолютно точное совпадение с названием (чтобы Avatar находил строго Avatar)
    exact_title_bonus = (df['title'].str.lower() == user_query_clean).astype(float)

    # 3. Объединяем все метрики в единый финальный скор (настраиваем веса)
    # - Название в приоритете (умножаем сходство на 3.0 + даем 5 баллов за подстроку + 10 за точный хит)
    # - Описание дополняет поиск (умножаем сходство на 1.0 + даем 1 балл за подстроку)
    final_search_scores = (
            (sim_title * 3.0) +
            (contains_title_bonus * 5.0) +
            (exact_title_bonus * 10.0) +
            (sim_overview * 1.0) +
            (contains_overview_bonus * 1.0)
        )

        # ИСПРАВЛЕНИЕ: Конвертируем в массив numpy (.values), чтобы [-1] означал конец массива, а не поиск индекса "-1"
    scores_array = final_search_scores.values
    winner_id = scores_array.argsort()[-1]

        # Если вообще ни одного совпадения нигде нет
    if scores_array[winner_id] == 0:
        return None

    target_movie = df.iloc[winner_id]

    # ---- ШАГ 2: ПОИСК РЕКОМЕНДАЦИЙ К НАЙДЕННОМУ ФИЛЬМУ ПО OVERVIEW ----
    target_vector = tfidf_overview_matrix[winner_id]
    all_recommendation_scores = linear_kernel(target_vector, tfidf_overview_matrix).flatten()

    # Сортируем индексы по убыванию сходства
    related_indices = all_recommendation_scores.argsort()[::-1]

    # Исключаем сам найденный фильм
    related_indices = [idx for idx in related_indices if idx != winner_id]

    # Берём топ-N рекомендаций
    top_indices = related_indices[:top_n]

    # Формируем датафрейм результатов
    recommendations = df[['title', 'vote_average']].iloc[top_indices].copy()
    recommendations['similarity_score'] = all_recommendation_scores[top_indices]

    return target_movie, recommendations


print("Сервер успешно запущен и готов к обработке запросов!")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=['GET', 'POST'])
def recommend_api():
    movie_title = None
    top_n = 10

    if request.method == 'GET':
        movie_title = request.args.get('title')
        top_n = int(request.args.get('limit', 10))
    elif request.method == 'POST':
        if request.is_json:
            data = request.json
            movie_title = data.get('title')
            top_n = int(data.get('limit', 10))
        else:
            movie_title = request.form.get('title')
            top_n = int(request.form.get('limit', 10))

    if not movie_title:
        return jsonify({"error": "Пожалуйста, укажите название фильма в параметре 'title'"}), 400

    # Вызываем гибридный поиск
    search_result = get_search_and_recommendations(movie_title, top_n=top_n)

    if search_result is None:
        return jsonify({"error": f"Фильм по запросу '{movie_title}' не найден в базе данных."}), 404

    target_movie, rec_df = search_result

    # 1. Создаем список результатов и ПЕРВЫМ вставляем сам найденный фильм (например, Аватар)
    final_results_list = []

    final_results_list.append({
        "title": str(target_movie['title']),
        "vote_average": float(target_movie['vote_average']),
        "similarity_score": 1.0,  # Сходство с самим собой всегда максимальное
        "is_target": True         # Флаг для фронтенда, что это именно искомый фильм
    })

    # 2. Добавляем в этот же список остальные рекомендованные фильмы
    for _, row in rec_df.iterrows():
        final_results_list.append({
            "title": str(row['title']),
            "vote_average": float(row['vote_average']),
            "similarity_score": round(float(row['similarity_score']), 4),
            "is_target": False
        })

    # Отдаем клиенту один общий список, где на 1-м месте стоит Аватар, а ниже — 10 рекомендаций
    return jsonify({
        "search_query": movie_title,
        "matched_movie_title": str(target_movie['title']),
        "results": final_results_list
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=False)
