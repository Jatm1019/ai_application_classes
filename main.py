from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
import image_class

app = FastAPI()

# ImageNetのクラスラベルを読み込む
imagenet_classes = ['dog','horse','elephant','butterfly','chicken','cat','cattle','sheep','spider','squirrel']
#with open("imagenet_classes.txt", "r") as f:
#    imagenet_classes = [line.strip() for line in f.readlines()]


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
        output = image_class.prediction_image(image)
        predicted_class = output.argmax().item()
        predicted_label = imagenet_classes[predicted_class]  # ← ラベル名を取得

        return {"predicted_class": predicted_class, "label": predicted_label}
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)