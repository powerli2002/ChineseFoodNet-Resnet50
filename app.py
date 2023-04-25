

import torch

import torchvision
from flask import Flask, render_template, request

from torchvision import transforms
from PIL import Image

import pandas

app = Flask(__name__)

# use_gpu = False
# device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
device = 'cpu'

@app.route('/predict', methods=['POST'])
def predict():
    classname = pandas.read_excel('class_names.xls')
    img = request.files['image']
    img = Image.open(img).convert('RGB')
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = torchvision.models.resnet50(pretrained=False).to(device)

    checkpoint = torch.load('model_data/resmodel-23-4000-0.4873.pt')
    model.load_state_dict(checkpoint['model'])
    model.eval() # 制定model.eval()固定dropout和BN层。
    # img = Image.open('./ChineseFoodNet/release_data/train/000/000000.jpg').convert('RGB')
    # img = Image.open('dataset/000000.jpg').convert('RGB')
    img = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52011104, 0.44459117, 0.30962785], std=[0.25595631, 0.25862494, 0.26925405])
    ])(img)
    img = img.reshape(1, 3, 128, 128).to(device)
    pre = model(img)

    _, predicted_top3 = pre.topk(3, dim=1)
    pretop3 = predicted_top3.tolist()

    pre_json = {}

    pre_json["top1-chname"] = classname["chname"][pretop3[0][0]]
    pre_json["top1-enname"] = classname["enname"][pretop3[0][0]]
    pre_json["top2-chname"] = classname["chname"][pretop3[0][1]]
    pre_json["top2-enname"] = classname["enname"][pretop3[0][1]]
    pre_json["top3-chname"] = classname["chname"][pretop3[0][2]]
    pre_json["top3-enname"] = classname["enname"][pretop3[0][2]]

    print(pre_json)





    return pre_json




if __name__ == '__main__':
    app.run()


