import torch
import argparse
import sys
sys.path.insert(0, '../2D-virtual-try-on')
from models.fc import Fc
from torchvision import  models



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/model.pt', help='model path')
    parser.add_argument('--device', type=str, default='cpu', help='device')
    return parser.parse_args()




args = parse_args()
checkpoint = torch.load(args.model_path)
onnx_model = args.model_path.replace('.pt', '.onnx')

model = models.resnet18(pretrained=True)
model.fc = Fc(in_features=model.fc.in_features, out_features=4)
model.load_state_dict(checkpoint['model'])
model.eval()

dummy_input = torch.zeros(1, 3, 256, 256)
output = model(dummy_input)

torch.onnx.export(model, dummy_input, onnx_model , verbose=True, input_names=['input'], output_names=['output'], opset_version=12)

print('ONNX conversion done')