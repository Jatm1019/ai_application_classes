from torchvision import models, transforms
import torch
from customize import CustomResNet18

# 事前学習済みのResNetモデルをロード
model = CustomResNet18()
model.load_state_dict(torch.load("model.pth"))
model.eval()    # 推論モード

# 画像をモデルが処理できる形に変換する関数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),     # (256,256)にresizeする
        transforms.CenterCrop(224), # 画像の中心を(224,224)で切り抜く
        transforms.ToTensor(),      # 画像をtensor形式に変換
    ])
    return transform(image).unsqueeze(0)    # .unsqueeze(0)バッチ次元数を追加

# 画像分類
def prediction_image(image):
    input_tensor = preprocess_image(image) 
    with torch.no_grad():
        output = model(input_tensor)
    return output