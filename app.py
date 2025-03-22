import joblib
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import asyncio
import json
import pickle
import warnings
import os

import timeit
from PIL import Image
import numpy as np
import cv2

from models.detr_model import DETR
from models.glpn_model import GLPDepth
from models.lstm_model import LSTM_Model
from models.predict_z_location_single_row_lstm import predict_z_location_single_row_lstm
from utils.processing import PROCESSING
from config import CONFIG

warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI(
    title="Real-Time WebSocket Image Processing API",
    description="API for object detection and depth estimation using WebSocket for real-time image processing.",
)

try:
    # Load models and utilities
    device = CONFIG['device']
    print("Loading models...")

    detr = DETR()  # Object detection model (DETR)
    print("DETR model loaded.")

    glpn = GLPDepth()  # Depth estimation model (GLPN)
    print("GLPDepth model loaded.")

    zlocE_LSTM = LSTM_Model()  # LSTM model for prediction (e.g., localization)
    print("LSTM model loaded.")

    zlocE_XGboost = xgb.Booster()
    zlocE_XGboost.load_model(CONFIG['xgboost_path'])
    print("XGboost model loaded.")

    lstm_scaler = pickle.load(open(CONFIG['lstm_scaler_path'], 'rb'))  # Load pre-trained scaler for LSTM
    print("LSTM Scaler loaded.")

    xgboost_scaler = joblib.load(CONFIG['xgboost_scaler_path'])
    print("XGboost Scaler loaded.")

    numerical_cols = joblib.load(CONFIG['numerical_cols_path'])

    processing = PROCESSING()  # Utility class for post-processing
    print("Processing utilities loaded.")

except FileNotFoundError as e:
    print(f"Error: A required file was not found. Details: {e}")

except KeyError as e:
    print(f"Error: Missing configuration key. Details: {e}")

except pickle.UnpicklingError as e:
    print(f"Error: Failed to load a pickle file. Details: {e}")

except joblib.externals.loky.process_executor.TerminatedWorkerError as e:
    print(f"Error: Joblib encountered an issue during model loading. Details: {e}")

except Exception as e:
    print(f"An unexpected error occurred. Details: {e}")



# Serve HTML documentation for the API
@app.get("/", response_class=HTMLResponse)
async def get_docs():
    """
    Serve HTML documentation for the WebSocket-based image processing API.
    The HTML file must be available in the same directory.
    Returns a 404 error if the documentation file is not found.
    """
    html_path = os.path.join(os.path.dirname(__file__), "api_documentation.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="api_documentation.html file not found", status_code=404)
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


@app.get("/try_page", response_class=HTMLResponse)
async def get_docs():
    """
    Serve HTML documentation for the WebSocket-based image processing API.
    The HTML file must be available in the same directory.
    Returns a 404 error if the documentation file is not found.
    """
    html_path = os.path.join(os.path.dirname(__file__), "try_page.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="try_page.html file not found", status_code=404)
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


# Function to decode the image received via WebSocket
async def decode_image(image_bytes):
    """
    Decodes image bytes into a PIL Image and returns the image along with its shape.

    Args:
        image_bytes (bytes): The image data received from the client.

    Returns:
        tuple: A tuple containing:
            - pil_image (PIL.Image): The decoded image.
            - img_shape (tuple): Shape of the image as (height, width).
            - decode_time (float): Time taken to decode the image in seconds.

    Raises:
        ValueError: If image decoding fails.
    """
    start = timeit.default_timer()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image")
    color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    img_shape = color_converted.shape[0:2]  # (height, width)
    end = timeit.default_timer()
    return pil_image, img_shape, end - start


# Function to run the DETR model for object detection
async def run_detr_model(pil_image):
    """
    Runs the DETR (DEtection TRansformer) model to perform object detection on the input image.

    Args:
        pil_image (PIL.Image): The image to be processed by the DETR model.

    Returns:
        tuple: A tuple containing:
            - detr_result (tuple): The DETR model output consisting of detections' scores and boxes.
            - detr_time (float): Time taken to run the DETR model in seconds.
    """
    start = timeit.default_timer()
    detr_result = await asyncio.to_thread(detr.detect, pil_image)
    end = timeit.default_timer()
    return detr_result, end - start


# Function to run the GLPN model for depth estimation
async def run_glpn_model(pil_image, img_shape):
    """
    Runs the GLPN (Global Local Prediction Network) model to estimate the depth of the objects in the image.

    Args:
        pil_image (PIL.Image): The image to be processed by the GLPN model.
        img_shape (tuple): The shape of the image as (height, width).

    Returns:
        tuple: A tuple containing:
            - depth_map (numpy.ndarray): The depth map for the input image.
            - glpn_time (float): Time taken to run the GLPN model in seconds.
    """
    start = timeit.default_timer()
    depth_map = await asyncio.to_thread(glpn.predict, pil_image, img_shape)
    end = timeit.default_timer()
    return depth_map, end - start


# Function to process the detections with depth map
async def process_detections(scores, boxes, depth_map):
    """
    Processes the DETR model detections and integrates depth information from the GLPN model.

    Args:
        scores (numpy.ndarray): The detection scores for the detected objects.
        boxes (numpy.ndarray): The bounding boxes for the detected objects.
        depth_map (numpy.ndarray): The depth map generated by the GLPN model.

    Returns:
        tuple: A tuple containing:
            - pdata (dict): Processed detection data including depth and bounding box information.
            - process_time (float): Time taken for processing detections in seconds.
    """
    start = timeit.default_timer()
    pdata = processing.process_detections(scores, boxes, depth_map, detr)
    end = timeit.default_timer()
    return pdata, end - start


# Function to generate JSON output for LSTM predictions
async def generate_json_output(data):
    """
       Predict Z-location for each object in the data and prepare the JSON output.

       Parameters:
       - data: DataFrame with bounding box coordinates, depth information, and class type.
       - ZlocE: Pre-loaded LSTM model for Z-location prediction.
       - scaler: Scaler for normalizing input data.

       Returns:
       - JSON structure with object class, distance estimated, and relevant features.
       """
    output_json = []
    start = timeit.default_timer()

    # Iterate over each row in the data
    for i, row in data.iterrows():
        # Predict distance for each object using the single-row prediction function

        distance = predict_z_location_single_row_lstm(row, zlocE_LSTM, lstm_scaler)

        # Create object info dictionary
        object_info = {
            "class": row["class"],  # Object class (e.g., 'car', 'truck')
            "distance_estimated": float(distance),  # Convert distance to float (if necessary)
            "features": {
                "xmin": float(row["xmin"]),  # Bounding box xmin
                "ymin": float(row["ymin"]),  # Bounding box ymin
                "xmax": float(row["xmax"]),  # Bounding box xmax
                "ymax": float(row["ymax"]),  # Bounding box ymax
                "mean_depth": float(row["depth_mean"]),  # Depth mean
                "depth_mean_trim": float(row["depth_mean_trim"]),  # Depth mean trim
                "depth_median": float(row["depth_median"]),  # Depth median
                "width": float(row["width"]),  # Object width
                "height": float(row["height"])  # Object height
            }
        }

        # Append each object info to the output JSON list
        output_json.append(object_info)

    end = timeit.default_timer()

    # Return the final JSON output structure, and time
    return {"objects": output_json}, end - start


# Function to process a single frame (image) in the WebSocket stream
async def process_frame(frame_id, image_bytes):
    """
    Processes a single frame (image) from the WebSocket stream. The process includes:
    - Decoding the image.
    - Running the DETR and GLPN models concurrently.
    - Processing detections and generating the final output JSON.

    Args:
        frame_id (int): The identifier for the frame being processed.
        image_bytes (bytes): The image data received from the WebSocket.

    Returns:
        dict: A dictionary containing the output JSON and timing information for each processing step.
    """
    timings = {}
    try:
        # Step 1: Decode the image
        pil_image, img_shape, decode_time = await decode_image(image_bytes)
        timings["decode_time"] = decode_time

        # Step 2: Run DETR and GLPN models in parallel
        (detr_result, detr_time), (depth_map, glpn_time) = await asyncio.gather(
            run_detr_model(pil_image),
            run_glpn_model(pil_image, img_shape)
        )
        models_time = max(detr_time, glpn_time)  # Take the longest time of the two models
        timings["models_time"] = models_time

        # Step 3: Process detections with depth map
        scores, boxes = detr_result
        pdata, process_time = await process_detections(scores, boxes, depth_map)
        timings["process_time"] = process_time

        # Step 4: Generate output JSON
        print("generate json")
        output_json, json_time = await generate_json_output(pdata)
        print(output_json)
        timings["json_time"] = json_time

        timings["total_time"] = decode_time + models_time + process_time + json_time

        # Add frame_id and timings to the JSON output
        output_json["frame_id"] = frame_id
        output_json["timings"] = timings

        return output_json

    except Exception as e:
        return {
            "error": str(e),
            "frame_id": frame_id,
            "timings": timings
        }


@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time image processing. Clients can send image frames to be processed
    and receive JSON output containing object detection, depth estimation, and predictions in real-time.

    - Handles the reception of image data over the WebSocket.
    - Calls the image processing pipeline and returns the result.

    Args:
        websocket (WebSocket): The WebSocket connection to the client.
    """
    await websocket.accept()
    frame_id = 0

    try:
        while True:
            frame_id += 1

            # Receive image bytes from the client
            image_bytes = await websocket.receive_bytes()

            # Process the frame asynchronously
            processing_task = asyncio.create_task(process_frame(frame_id, image_bytes))
            result = await processing_task

            # Send the result back to the client
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print(f"Client disconnected after processing {frame_id} frames.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await websocket.close()
