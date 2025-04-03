from torchvision import models, transforms
import torch

# 事前学習済みのResNetモデルをロード
model = models.resnet18(pretrained=True)
model.eval()

# 画像をモデルが処理できる形に変換する関数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# 画像分類
def prediction_image(image):
    input_tensor = preprocess_image(image) 
    with torch.no_grad():
        output = model(input_tensor)
    return output