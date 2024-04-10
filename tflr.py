from joblib import load

lr_path = "./lr_classifier.joblib"
vectorizer_path = "./tfidf_vectorizer.joblib"

vectorizer=""
lr = ""

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
        access_key="YCAJEFaglepXGGXvwlyeXZPOm",
        secret_key="YCPd-IJkxTVo3tvmVgEI4dyIGwoComU77MszPrhI"
    )


def init():
    global vectorizer, lr
    global classifiers
    genres = ['genre_rock_classifier.joblib', 'genre_pop_classifier.joblib', 'genre_misc_classifier.joblib',
              'genre_rap_classifier.joblib', 'genre_rb_classifier.joblib', 'genre_country_classifier.joblib']
    for genre in genres:
        load_func(genre)
        classifier = load(genre)
        classifiers[genre] = classifier
    load_func("tfidf_vectorizer.joblib")
    vectorizer = load(vectorizer_path)

def predict(song_text):
    global vectorizer
    X = [song_text]
    X = vectorizer.transform(X)
    mas = []
    for genre, classifier in classifiers.items():
        prediction = classifier.predict_proba(X)[0, 1]
        if(prediction>=0.8):
            mas.append(genre)
    return mas