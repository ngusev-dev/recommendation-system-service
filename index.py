from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Чтобы русский текст в JSON отображался корректно

print("Инициализация сервиса и загрузка данных...")

# Загружаем датасет локально
csv_path = 'tmdb_5000_movies.csv'
try:
    df = pd.read_csv(csv_path)
    print(f"Успешно загружено фильмов: {len(df)}")
except FileNotFoundError:
    print(f"Критическая ошибка: Файл {csv_path} не найден в текущей директории.")
    raise

# Удаляем строки, где текстовое описание (overview) отсутствует
df['overview'] = df['overview'].fillna('')

# Применяем векторизатор TF-IDF (из вашего алгоритма)
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf.fit_transform(df['overview'])
print(f"Размерность матрицы признаков: {tfidf_matrix.shape}")

# Вычисляем матрицу сходства между всеми фильмами
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Создаем обратный индекс для быстрого поиска индекса фильма по его названию
indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()


# Ваша базовая функция рекомендаций (адаптированная для веб-выдачи)
def get_advanced_recommendations(title, cosine_sim=cosine_sim, df=df, indices=indices, top_n=5):
    title_lower = title.lower()

    if title_lower not in indices:
        return None  # Возвращаем None, если фильм не найден

    # Получаем индекс фильма
    idx = indices[title_lower]

    # Если названию соответствует несколько индексов, берем первый
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Получаем оценки сходства для всех фильмов с выбранным
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Сортируем по убыванию сходства
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Берем топ-N фильмов (исключая сам этот фильм, который стоит на 0-м месте)
    sim_scores = sim_scores[1:top_n+1]

    # Получаем индексы и score
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    # Формируем результат
    result = df[['title', 'vote_average']].iloc[movie_indices].copy()
    result['similarity_score'] = scores

    return result

print("Сервер успешно запущен и готов к обработке запросов!")

# ==========================================
# МАРШРУТЫ (ВЕБ-ИНТЕРФЕЙС И API)
# ==========================================

# Главная страница, которая загружает интерфейс
@app.route("/")
def home():
    return render_template("index.html")


# API-endpoint для выдачи результатов
@app.route("/recommend", methods=['GET', 'POST'])
def recommend_api():
    movie_title = None
    top_n = 5
    
    # Считываем параметры из GET или POST запросов
    if request.method == 'GET':
        movie_title = request.args.get('title')
        top_n = int(request.args.get('limit', 5))
    elif request.method == 'POST':
        if request.is_json:
            data = request.json
            movie_title = data.get('title')
            top_n = int(data.get('limit', 5))
        else:
            movie_title = request.form.get('title')
            top_n = int(request.form.get('limit', 5))
            
    if not movie_title:
        return jsonify({"error": "Пожалуйста, укажите название фильма в параметре 'title'"}), 400
        
    # Вызываем ваш алгоритм
    recommendations_df = get_advanced_recommendations(movie_title, top_n=top_n)
    
    # Проверяем, нашелся ли фильм
    if recommendations_df is None:
        return jsonify({"error": f"Фильм '{movie_title}' не найден в базе данных."}), 404
        
    # Превращаем DataFrame в список словарей для JSON ответа
    recommendations_list = []
    for _, row in recommendations_df.iterrows():
        recommendations_list.append({
            "title": str(row['title']),
            "vote_average": float(row['vote_average']),
            "similarity_score": round(float(row['similarity_score']), 4)
        })
        
    return jsonify({
        "requested_movie": movie_title,
        "recommendations": recommendations_list
    })


if __name__ == "__main__":
    # Запускаем локальный веб-сервер на порту 5002
    app.run(host='0.0.0.0', port=5002, debug=False)