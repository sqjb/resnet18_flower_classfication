import argparse
import json

import torch
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description="flower classification by resnet18")
parser.add_argument("--image", help='image path', default='datas/test.png')
parser.add_argument("--weights", help='resnet model path', default="./resnet18.pth")
args = parser.parse_args()

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def infer(path: str, weights: str):
    # label
    with open("labels.json", "r") as f:
        label = json.load(f)

    model = torch.load(weights)
    model.to(device)
    model.eval()

    image = Image.open(path)
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = trans(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    pred = model.forward(image)
    cls, conf = torch.max(pred, 1)[1], torch.softmax(pred, 1).max().item()

    name = label[str(cls[0].item())]
    print(name, conf)


if __name__ == '__main__':
    infer(args.image, args.weights)
