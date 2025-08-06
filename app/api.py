# app/api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from app.utils import detect_and_classify

app = FastAPI()

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def root():
    return {"message": "PPE Vision 360 API is running!"}

@app.post("/check_compliance")
async def check_compliance(file: UploadFile = File(...)):
    # Save uploaded file
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run Detection & Classification
    result = detect_and_classify(file_location)

    # Clean up uploaded file (optional)
    os.remove(file_location)

    return JSONResponse(content=result)
