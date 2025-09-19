# Prepare propagated planar masks

Note that with using automated masks, if you were to use a test set, you need to label it manually anyways with Flow 2.

## Flow 1: Training with automated masks for Scannet++


### Step 1. Run PlaneRecNet

1. Clone the repo and change the inference file

```
git clone https://github.com/EryiXie/PlaneRecNet.git

cd PlaneRecNet

wget -O simple_inference.py https://gist.githubusercontent.com/mtaktash/7bb20e5e77b3fa481aa37d14e9a4a07c/raw/40639930cda314393803da7be748f8c2dfd63fef/simple_inference.py

```
2. Use PlaneRecNet README to create environment and download checkpoints into `PlaneRecNet/weights`

3. Run PlaneRecNet

```
conda activate prn_test

python3 simple_inference.py --config=PlaneRecNet_101_config --trained_model=weights/PlaneRecNet_101_9_125000.pth --score_threshold=0.5 --images=input_folder:output_folder
```

### Step 2: Run SAM-2 video propagation

1. Create a new conda environment
```
conda create -n sam2 python=3.12
conda activate sam2
pip install opencv-python matplotlib scipy
pip install 'git+https://github.com/facebookresearch/sam2.git'
pip install huggingface-hub
```

2. Run propagation
```
cd sam2

python run_sam2_mask_propagation.py -i "input frames path (jpeg)" -t "train test lists path" -o "output path" -p "planerecnet output folder from Step 1"
```


## Flow 2: Manual labelling and propagation with SAM-2

1. Clone a UI for SAMv2
```
git clone https://github.com/mtaktash/sam-ui.git
```
2. Follow the README to label your own data