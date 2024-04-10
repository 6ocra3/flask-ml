from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tflr
app = FastAPI()
tflr.init()
class TextItem(BaseModel):
    text: str

@app.get("/")
def process_text():
    return "Hello World"

@app.post("/process-text/")
async def process_text(item: TextItem):
    print(tflr.predict(item.text))
    prediction = tflr.predict(item.text)
    return prediction