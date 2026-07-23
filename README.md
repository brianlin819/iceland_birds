# iceland_birds
Model and data to classify Rock Ptarmigans in Iceland

## SETUP

1. Install [Conda](http://conda.io/)

2. create a virtual environment and install the requirements

3. Install Pytorch. You will need to install the version of pytorch that fits with your machine. Most people should install it from [here](https://pytorch.org/get-started/locally/) but if you are planning on using an intel gpu, install it from [here](https://docs.pytorch.org/docs/2.13/notes/get_start_xpu.html). If downloading with CUDA, ROCm, or Intel XPU, make sure you have the proper drivers installed

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

## (OPTIONAL) Adding Data
If you would like to train on your own dataset rather than the ones provided, delete the folders inside "datasets" and add your own folder with your data and annotations. Then change the image path in "detect_and_segment.py" to the path of your folder. Finally run "detect_and_segment.py" and then "pad_images.py".

```bash
python prepare_data_scripts/detet_and_segment.py
```

```bash
python prepare_data_scripts/pad_images.py
```

**Note:** My annotations for my project have already been split for training/validation/testing. If yours aren't, you may have to adjust dataset.py and train.py to account for that.