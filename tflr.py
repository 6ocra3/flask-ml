from joblib import load
import numpy
import json
import os
import csv

with open('config.json') as config_file:
    config = json.load(config_file)

access_key = config['access_key']
secret_key = config['secret_key']

vectorizer_path = "tfidf.joblib"
classifiers = dict()
import boto3
from botocore.exceptions import NoCredentialsError


def download_file_from_yandex_cloud(bucket_name, object_name, file_path, endpoint_url, access_key, secret_key):
    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        s3.download_file(bucket_name, object_name, file_path)
        print(f"Файл {object_name} успешно скачан и сохранён как {file_path}.")
    except NoCredentialsError:
        print("Ошибка аутентификации. Проверьте ваши ключи доступа.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")


def load_func(name):
    download_file_from_yandex_cloud(
        bucket_name="makar-airflow-s3",
        object_name=f"{name}",
        file_path=f"./{name}",
        endpoint_url="https://storage.yandexcloud.net",
        access_key=access_key,
        secret_key=secret_key
    )


def init():
    global vectorizer, lr
    global classifiers

    reports_file = 'reports.csv'

    if os.path.exists(reports_file):
        os.remove(reports_file)
        print(f"File {reports_file} removed.")

    with open(reports_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['lem_lyrics', 'genre'])

    genres = ['f_genre_rock_classifier.joblib', 'f_genre_pop_classifier.joblib', 'f_genre_misc_classifier.joblib',
              'f_genre_rap_classifier.joblib', 'f_genre_rb_classifier.joblib', 'f_genre_country_classifier.joblib']
    for genre in genres:
        load_func(genre)
        classifier = load(genre)
        classifiers[genre] = classifier
    load_func("tfidf.joblib")
    vectorizer = load(vectorizer_path)


def predict(song_text):
    global vectorizer
    X = vectorizer.transform([song_text])
    genres = ["rock", "pop", "misc", "rap", "rb", "country"]
    mas = []
    ans = []
    for genre, classifier in classifiers.items():
        prediction = classifier.predict_proba(X)[0, 1]
        mas.append(prediction)
    std = numpy.std(mas)
    maxi = max(mas)
    for i in range(len(mas)):
        if (mas[i] >= maxi - std):
            ans.append(genres[i])
    return ans
