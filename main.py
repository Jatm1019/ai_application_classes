from fastapi import FastAPI, File, UploadFile
import io
from torchvision import models, transforms
import torch
from PIL import Image

app = FastAPI()

# 事前学習済みのResNetモデルをロード
model = models.resnet18(pretrained=True)
model.eval()

# ImageNetのクラスラベルを読み込む
imagenet_classes = []
with open("imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# 画像をモデルが処理できる形に変換する関数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/hello")
def say_hello():
    return {"message": "Hello, World!"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        input_tensor = preprocess_image(image)  # 修正
        with torch.no_grad():
            output = model(input_tensor)
        predicted_class = output.argmax().item()
        predicted_label = imagenet_classes[predicted_class]  # ← ラベル名を取得

        return {"predicted_class": predicted_class, "label": predicted_label}
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)