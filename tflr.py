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

def init():
    global vectorizer, lr
    download_file_from_yandex_cloud(
        bucket_name="makar-airflow-s3",
        object_name="lr_classifier.joblib",
        file_path="./lr_classifier.joblib",
        endpoint_url="https://storage.yandexcloud.net",
        access_key="YCAJEFaglepXGGXvwlyeXZPOm",
        secret_key="YCPd-IJkxTVo3tvmVgEI4dyIGwoComU77MszPrhI"
    )
    download_file_from_yandex_cloud(
        bucket_name="makar-airflow-s3",
        object_name="tfidf_vectorizer.joblib",
        file_path="./tfidf_vectorizer.joblib",
        endpoint_url="https://storage.yandexcloud.net",
        access_key="YCAJEFaglepXGGXvwlyeXZPOm",
        secret_key="YCPd-IJkxTVo3tvmVgEI4dyIGwoComU77MszPrhI"
    )
    vectorizer = load(vectorizer_path)
    lr = load(lr_path)

def predict(song_text):
    global vectorizer, lr
    X = [song_text]
    X = vectorizer.transform(X)
    ans = lr.predict(X)
    return ans