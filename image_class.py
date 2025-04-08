from torchvision import models, transforms
import torch
import customize

# import mlflow
# import mlflow.sklearn
# from sklearn.metrics import accuracy_score

# MLflowの設定
# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("image_classification_experiment")

# 事前学習済みのResNetモデルをロード
model = customize.CustomResNet18()
model.eval()

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
        output = model.forward(input_tensor)
    print(output)
    return output