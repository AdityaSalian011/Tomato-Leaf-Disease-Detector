from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tomato_leaf_app import apply_transformer, load_model, predict
from PIL import Image
from io import BytesIO

#Step1: instantiate FastAPI
app = FastAPI()

#Step2: Add CORSMiddleware and origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

#Step3: Load our model once at startup
model = load_model()

#Step4: Create a predict route
@app.post('/predict')
async def predict_route(image: UploadFile = File(...)):
    """Frontend sends image as bytes
        We make prediction on the image and send class name and confidence level."""
    #Step5: Convert byte image to RGB Image
    img = Image.open(BytesIO(await image.read())).convert('RGB')
    
    #Step6: Convert PIL image (H,W,C) into Tensor(B,C,H,W)
    tensor = apply_transformer(img)

    #Step7: Predict class name and confidence percentage
    class_name, confidence = predict(model=model, image=tensor)

    #Step8: Return data back to frontend
    return {
        'class': class_name,
        'confidence': float(confidence)
    }

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)