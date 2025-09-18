from fastapi import FastAPI, File, UploadFile, Form
from openai import OpenAI
import shutil
from pathlib import Path
import json


app = FastAPI()

# OpenAI API 클라이언트 설정
client = OpenAI(api_key='API_KEY')


# 파일 저장 경로
UPLOAD_DIRECTORY = Path("uploaded_files")
UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    target_text: str = Form(...)
):
    # 저장할 파일 경로 설정
    file_path = UPLOAD_DIRECTORY / file.filename
    
    # 파일 저장
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # OpenAI Whisper API를 사용하여 파일을 텍스트로 변환
    with file_path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            file=audio_file,
            model='whisper-1',
            response_format='text',
            temperature=0.0
        )
    
    transcribed_text = result.strip()  # 공백 제거
    print(f"Transcribed text: {transcribed_text}")

    # 목표 텍스트와 비교
    if target_text in transcribed_text:
        message = "정답"
    else:
        message = "오답"

    response_data = {
        "filename": file.filename,
        "message": message,
        "transcribed_text": transcribed_text
    }

    # JSON 파일로 저장
    json_file_path = UPLOAD_DIRECTORY / f"{file.filename}.json"
    with json_file_path.open("w", encoding="utf-8") as json_file:
        json.dump(response_data, json_file, ensure_ascii=False, indent=4)

    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
