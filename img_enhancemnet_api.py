from fastapi import FastAPI, File, UploadFile, Response, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from enum import Enum
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI()
templates = Jinja2Templates(directory="./templates")

class Operation(str, Enum):
    smoothen = "smoothen"
    sharpen = "sharpen"
    edge_detect = "edge_detect"
    
def convert_byte_to_array(image):
    return np.array(Image.open(BytesIO(image)))

def convert_array_to_byte(image_arr):
    image_arr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)              # array in RGB --> GRB so that cv2 doesnt reverse the color
    success, image_byte = cv2.imencode('.jpg', image_arr)
    return image_byte.tobytes()
    
def apply_smoothing(img):
    # Gaussian blur
    smoothed_image = cv2.GaussianBlur(img, (3, 3), 1)
    return smoothed_image

def apply_sharpening(img):
    # High boost filtering
    smooth_img = apply_smoothing(img)
    sharpened_img = cv2.addWeighted(img, 2, smooth_img, -1, 0)   
    return sharpened_img

def apply_edge_detect(img):
    # Canny filtering
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_img = cv2.Canny(gray_img, 100, 200, 3)
    return edge_img

# Home page
@app.get("/")
async def home(request: Request):
    links = [
        {"description": "Overview", "url": "/docs"},
        {"description": "Upload Image", "url": "/upload", "method": "GET"},
        {"description": "Image Enhancement", "url": "/enhance", "method": "GET"}
    ]
    return templates.TemplateResponse("home.html", {"request": request, "links": links})
 
@app.post("/upload")
async def upload_file(request: Request, image: UploadFile = File(...)):
    global image_uploaded
    image_uploaded = await image.read()
    return templates.TemplateResponse("upload.html", {"request": request, "file_name": image.filename})

@app.post("/enhance")
async def processed_image(
    operation: Operation = Operation.smoothen
):  
    # Select operation
    if operation == Operation.smoothen:
        return RedirectResponse(url="/smoothen")
    elif operation == Operation.sharpen:
        return RedirectResponse(url="/sharpen")
    elif operation == Operation.edge_detect:
        return RedirectResponse(url="/edge_detect")
    else:
        return {"error": "Invalid Operation!"}
    
         
@app.post("/enhance/smoothen")
async def smoothen():
    img_arr = convert_byte_to_array(image_uploaded)
    processed_image = apply_smoothing(img_arr)
    img_byte = convert_array_to_byte(processed_image)
    return Response(img_byte, media_type='image/jpg')

@app.post("/enhance/sharpen")
async def sharpen():
    img_arr = convert_byte_to_array(image_uploaded)
    processed_image = apply_sharpening(img_arr)
    img_byte = convert_array_to_byte(processed_image)
    return Response(img_byte, media_type='image/jpg')

@app.post("/enhance/edge_detect")
async def edge_detect():
    img_arr = convert_byte_to_array(image_uploaded)
    processed_image = apply_edge_detect(img_arr)
    img_byte = convert_array_to_byte(processed_image)
    return Response(img_byte, media_type='image/jpg')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    