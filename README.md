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
## Running
1.     
**IMPORTANT:** If you are not running this on intel XPU, you will have to go into the code and manually change xpu into the device you are using (cuda, cpu, etc.) in *configs*, *util.py*, and *train.py* 

2. 
Run train.py by running:
```bash
python bird_classifier/train.py --config configs/exp_efficientnet.yaml
```
If you want to graph the training and validation loss/accuracies, run plot_results.py:

```bash
python bird_classifier/plot_results.py
```

Graphs for the test set will only be plotted once training has finished and gone through all the epochs set in configs.


If training the model will take too long, a couple plots of the most recent run can be found in the graphs folder. (The results aren't very good yet)