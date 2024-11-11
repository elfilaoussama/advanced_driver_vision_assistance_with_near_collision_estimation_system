import os
import json
import pickle
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import cv2
from models.detr_model import DETR
from models.glpn_model import GLPDepth
from models.lstm_model import LSTM_Model
from utils.JSON_output import generate_output_json
from utils.processing import PROCESSING
from config import CONFIG

app = FastAPI(
    title="WebSocket Image Upload API",
    description="API for uploading images via WebSocket and receiving object detection and depth estimation results."
)

# Load the models and configurations
device = CONFIG['device']
print("Loading models...")

try:
    detr = DETR()
    print("DETR model loaded.")
    
    glpn = GLPDepth()
    print("GLPDepth model loaded.")
    
    zlocE = LSTM_Model()
    print("LSTM model loaded.")
    
    scaler = pickle.load(open(CONFIG['lstm_scaler_path'], 'rb'))
    print("Scaler loaded.")
    
    processing = PROCESSING()
    print("Processing utilities loaded.")

except Exception as e:
    print(f"Error loading models or utilities: {e}")

# Serve the HTML documentation
@app.get("/", response_class=HTMLResponse)
async def get_docs():
    html_path = os.path.join(os.path.dirname(__file__), "api_documentation.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="docs.html file not found", status_code=404)
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())

# Run DETR detection in a separate thread
async def run_detr(pil_image):
    return await asyncio.to_thread(detr.detect, pil_image)

# Run GLPN depth estimation in a separate thread
async def run_glpn(pil_image, img_shape):
    return await asyncio.to_thread(glpn.predict, pil_image, img_shape)

# WebSocket endpoint for receiving images and returning predictions
@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize frame counter
    frame_counter = 0
    
    try:
        while True:
            try:
                # Increment frame counter for each new frame
                frame_counter += 1
                
                # Receive raw bytes (image data)
                image_bytes = await websocket.receive_bytes()
                
                # Convert bytes to a NumPy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                
                # Decode the image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Failed to decode image")

                # Resize the image (if necessary) and convert to RGB
                color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(color_converted)
                img_shape = color_converted.shape[0:2]  # (height, width)

                # Run DETR and GLPN in parallel
                detr_result, depth_map = await asyncio.gather(
                    run_detr(pil_image),
                    run_glpn(pil_image, img_shape)
                )
                
                # Unpack the DETR detection results
                scores, boxes = detr_result

                # Process bounding boxes and overlap them with depth map
                pdata = processing.process_detections(scores, boxes, depth_map, detr)

                # Generate the output JSON
                output_json = generate_output_json(pdata, zlocE, scaler)

                # Include frame number in the JSON
                output_json['frame_number'] = frame_counter

                # Send the output back to the client (JSON result)
                await websocket.send_text(json.dumps(output_json))

            except ValueError as ve:
                await websocket.send_text(f"Image processing error: {ve}")
            except Exception as e:
                await websocket.send_text(f"Unexpected error: {str(e)}")

    except WebSocketDisconnect:
        print(f"Client disconnected. Total frames processed: {frame_counter}")
        frame_counter = 0  # Reset frame counter when connection closes
    finally:
        await websocket.close()