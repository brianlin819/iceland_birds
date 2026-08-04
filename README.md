# iceland_birds
Model and data to classify Rock Ptarmigans in Iceland

## SETUP

1. Clone the repository

2. create a virtual environment and activate it

3. Install Pytorch. You will need to install the version of pytorch that fits with your machine. Most people should install it from [here](https://pytorch.org/get-started/locally/) but if you are planning on using an intel gpu, install it from [here](https://docs.pytorch.org/docs/2.13/notes/get_start_xpu.html). If downloading with CUDA, ROCm, or Intel XPU, make sure you have the proper drivers installed

### Setup for CPU (if you have no GPU. Recommended if just testing)
```bash
git clone https://github.com/brianlin819/iceland_birds
cd iceland_birds
python3 -m venv iceland_birds
source iceland_birds/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```


### Pytorch Installation for Intel GPU
Download drivers
```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
sudo apt-get install -y intel-media-va-driver-non-free libmfx-gen1.2 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo
sudo apt-get install -y libze-dev intel-ocloc
```

Setup with Intel GPU
```bash
git clone https://github.com/brianlin819/iceland_birds
python -m venv iceland_birds
source iceland_birds/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install -r requirements.txt
```
## Running

Run train.py by running:
```bash
python bird_classifier/train.py --config configs/exp_efficientnet.yaml
```
If you want to graph the training and validation loss/accuracies (MUST HAVE RAN TRAINING FOR AT LEAST ONE EPOCH), run plot_results.py:

```bash
python bird_classifier/plot_results.py
```

Graphs for the test set will only be plotted once training has finished and gone through all the epochs set in configs.


If training the model will take too long, a couple plots of the most recent run can be found in the graphs folder. The most recent results overfit pretty severely. But even so, the accuracy for the test set and validation set are significantly better than before. Will continue working on reducing overfitting...

## Creating the Paper and Presentation
To view the presentation, download the BrianLin_ClassifyingRockPtarmiganAgeAndSex_slides(1).odp file.

To view the paper, 
```bash
sudo apt-get install texlive-latex-base
sudo apt update
sudo apt install texlive-full
cd overleaf_paper_source/
pdflatex iceland_birds.tex
bibtex iceland_birds
pdflatex iceland_birds.tex
```
Download the iceland_birds.pdf paper that shows up in the overleaf_paper_source folder. 

OR

Go to overleaf.com and create a blank project. Upload the .tex, .pngs, and .bib and delete the default placeholder file. Then open the .tex file and compile.

## (OPTIONAL) Adding Data
If you would like to train on your own dataset rather than the ones provided, delete the folders inside "datasets" and add your own folder with your data and annotations. Then change the image path in "detect_and_segment.py" to the path of your folder. Finally run "detect_and_segment.py" and then "pad_images.py".

```bash
python prepare_data_scripts/detet_and_segment.py
```

```bash
python prepare_data_scripts/pad_images.py
```

**Note:** My annotations for my project have already been split for training/validation/testing. If yours aren't, you may have to adjust dataset.py and train.py to account for that.