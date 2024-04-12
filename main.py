from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.responses import FileResponse
import os
import tflr
app = FastAPI()
tflr.init()
class TextItem(BaseModel):
    text: str

@app.get("/")
def process_text():
    song = "rap nigga money babys"
    prediction = tflr.predict(song)
    return prediction

@app.post("/process-text/")
async def process_text(item: TextItem):
    print(tflr.predict(item.text))
    prediction = tflr.predict(item.text)
    return prediction

@app.get("/get_reports/")
async def download_file():
    file_path = f"./reports.csv"
    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    # Возвращаем файл как ответ
    return FileResponse(file_path, filename="reports.csv")