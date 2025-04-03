import requests

url = "http://localhost:8000/predict"
image_path = "test.jpg"  # 任意の画像ファイル

with open(image_path, "rb") as image_file:
    files = {"file": image_file}
    response = requests.post(url, files=files)

print(response.json())  # 予測結果を表示