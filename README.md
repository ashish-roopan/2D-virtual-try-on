# Setup
```
git clone git@github.com:ashish-roopan/2D-virtual-try-on.git
cd 2D-virtual-try-on.git
pip install -r requirements.txt
```

# Inference.

#### For a single image folder .Example samples/1029/
```
python demo/onnx_demo.py --model_path ResNet__1037__0.002__0.010.onnx --data_dir samples/1029
```
#### For multile image folders in a dataset dir .Example dataset/test_set/*
```
python demo/onnx_demo_multiple_images.py --model_path checkpoints/ResNet__1037__0.002__0.010.onnx --data_dir dataset/test_set/
```


# Train

#### From scratch
```
python main.py  --batch_size 10 --debug True --num_epochs 1000 
```
#### Resume from a checkpoint
```
python main.py  --batch_size 10 --debug True --num_epochs 1000 --load_model True --model_path checkpoints/ResNet__200__0.060__0.099.pt
```

# Convert to onnx model
```
python utils/convert2onnx.py --model_path checkpoints/ResNet__0__20.203__1.678.pt
```
