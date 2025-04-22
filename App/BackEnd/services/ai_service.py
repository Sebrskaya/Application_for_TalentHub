import requests
from fastapi import HTTPException

TORCHSERVE_URL = "http://localhost:8080/predictions/model_name"


def analyze_video(video_path: str):
    files = {"file": open(video_path, "rb")}
    response = requests.post(TORCHSERVE_URL, files=files)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Ошибка анализа видео")

    return response.json()