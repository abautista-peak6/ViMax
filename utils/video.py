import logging
from urllib.parse import quote
import requests
from tenacity import retry


@retry
def download_video(url, save_path):
    try:
        logging.info(f"Downloading video from {url} to {save_path}")

        if url.startswith("gs://"):
            response = _download_gcs(url)
        else:
            response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
    
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Video downloaded successfully to {save_path}")
    
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        raise e


def _download_gcs(gcs_uri: str):
    from google.auth.transport.requests import AuthorizedSession
    import google.auth

    bucket_and_path = gcs_uri.removeprefix("gs://")
    bucket, object_name = bucket_and_path.split("/", 1)
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = AuthorizedSession(credentials)
    url = (
        "https://storage.googleapis.com/storage/v1/b/"
        f"{quote(bucket, safe='')}/o/{quote(object_name, safe='')}?alt=media"
    )
    return session.get(url, stream=True)
