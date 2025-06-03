import base64, os, uuid, shutil, cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import asyncio


VIDEO_UPLOAD_DIR = "uploaded_videos"
ANNOTATED_DIR = "annotated_videos"

os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)


app = FastAPI()


app.mount("/annotated_videos", StaticFiles(directory=ANNOTATED_DIR), name="annotated_videos")

# Allow all CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = YOLO("yolov8n.pt")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            if "," not in data:
                continue

            image_data = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(image_data, np.uint8)

            if np_arr.size == 0:
                print("Received empty image buffer.")
                continue

            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Could not decode frame.")
                continue

            results = model(frame)[0]
            annotated_frame = results.plot()

            _, buffer = cv2.imencode(".jpg", annotated_frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")

    except WebSocketDisconnect:
        print("Client disconnected")


@app.post("/image")
async def detect_image(request: Request):
    data = await request.json()
    base64_image = data.get("image")

    if not base64_image or "," not in base64_image:
        return JSONResponse(content={"error": "Invalid image data"}, status_code=400)

    try:
        image_data = base64.b64decode(base64_image.split(",")[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(frame)[0]
        annotated = results.plot()

        _, buffer = cv2.imencode(".jpg", annotated)
        encoded_result = base64.b64encode(buffer).decode("utf-8")
        return {"annotated": f"data:image/jpeg;base64,{encoded_result}"}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/video")
async def process_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    input_path = os.path.join(VIDEO_UPLOAD_DIR, f"{video_id}_{file.filename}")
    output_path = os.path.join(ANNOTATED_DIR, f"{video_id}_annotated.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return JSONResponse(content={"error": "Failed to read video"}, status_code=400)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25  # Fallback if FPS is 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec, browser compatible
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            annotated = results.plot()
            out.write(annotated)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        cap.release()
        out.release()

    # Return relative path to static file
    video_url = f"http://localhost:8000/{output_path.replace(os.sep, '/')}"
    return {"video": video_url}

@app.get("/")
def root():
    return FileResponse("index.html")