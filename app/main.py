from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import numpy as np
import cv2
import io

from dataset.Dataloader import testloader
from classification_model.model import *

BASE_DIR = '/media/victor/851aa2dd-6b93-4a57-8100-b5253aa4eedd/cursos/checkpoint_model_microservice/epoch=3-step=49999.ckpt'
CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

app = FastAPI()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# pydantic models

class ImageClassifier(BaseModel):
    plane: float
    car: float
    bird: float
    cat: float
    deer: float
    dog: float
    frog: float
    horse: float
    ship : float
    truck: float
    
def pass_image():
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    return images, labels

def predict():
    model = Mlmodel.load_from_checkpoint(BASE_DIR, map_location=device, strict=False)
    model.eval()
    return model

def transform_image(image):
    image_np = image.numpy() #Convert the tensor to numpy and host it on the cpu
    image = np.transpose(image_np, (0, 2, 3, 1))
    return image

# routes
@app.get("/ping")
def pong():
    return {"ping": "pong!"}

@app.get("/predict")
async def predict_image():
    images, labels = pass_image()
    model = predict()
    logit = model(images)
    _, prediction = torch.max(logit, dim=1)
    output = []
    for idx in prediction:
        output.append(CLASSES[idx])
    response_object = {'class predicted': output}
    im = transform_image(images)
    res, im_png = cv2.imencode('.png', im)
    return response_object, StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")



