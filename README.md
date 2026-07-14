# iceland_birds
Model and data to classify Rock Ptarmigans in Iceland

## SETUP

1. Install [Conda](http://conda.io/)

2. Install Pytorch. You will need to install the version of pytorch that fits with your machine. Most people should install it from [here](https://pytorch.org/get-started/locally/) but if you are planning on using an intel gpu, install it from [here](https://docs.pytorch.org/docs/2.13/notes/get_start_xpu.html). If downloading with CUDA, ROCm, or Intel XPU, make sure you have the proper drivers installed

Example of Setup
```bash
conda create -n iceland_birds
conda activate iceland_birds
conda install pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
```
