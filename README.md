# CALIMAR-GAN: An unpaired mask-guided attention network for metal artifact reduction in CT scans

R.M. Scardigno, A. Brunetti, P.M. Marvulli, R. Carli, M. Dotoli, V. Bevilacqua, D. Buongiorno

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/roberto722/calimar-gan.git
cd CalimarGAN
```

2. (Recommended) Create a Conda environment with Python 3.10:

```bash
conda create -n calimar-gan python=3.10
conda activate calimar-gan
```

3. Install dependencies (with CUDA support for Astra Toolbox), please follow the instructions in this exact order:

```bash
# Install astra-toolbox via conda (required for CUDA on Windows), via pip works only on Linux.
conda install astra-toolbox -c astra-toolbox -c nvidia

# Install PyTorch with CUDA 11.8 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional Python dependencies
pip install opencv-python
pip uninstall numpy==2.2.6

pip install https://github.com/odlgroup/odl/archive/master.zip
pip install dominate tqdm matplotlib
```

> Make sure you activate your Conda environment before running any scripts.

---

## 🚀 How to Use

### 🏋️‍♂️ Training

To start training a model:

```bash
python run.py train configs/train_config.json
```

### 🧪 Testing / Inference

To evaluate or generate output from a trained model:

```bash
python run.py test configs/test_config.json
```

> Trained model checkpoints must exist in `checkpoints/<experiment_name>/`.

---

## 🗂️ Dataset Structure

The dataset structure expected by `UnalignedCalimarganDataset` differs slightly for training and testing. In the official implementation, the training data comprises both simulated and real samples. However, the training can also be done using only one type of data.

The original dataset website is: [http://spineweb.digitalimaginggroup.ca](http://spineweb.digitalimaginggroup.ca)

### 🏋️ Training (Simulated + Real)

```
<dataroot>/
├── A/
│   ├── 2539494/              # Simulated samples
│   │   ├── 69/
│   │   │   ├── 1_HU.mat
│   │   │   ├── 1_mask.mat
│   │   │   ├── ...
│   ├── 2495819/              # Real samples
│   │   ├── 1/
│   │   │   ├── art_HU.mat
│   │   │   └── art.mat
└── B/
    ├── 2539494/              
    │   ├── 69/
    │   │   └── gt.mat
    │   │
    └── ...

```

- **Simulated samples** include multiple `*_HU.mat` and `*_mask.mat` files per slice (1 to 25).
- **Real samples** include `art_HU.mat`, used as fallback when a numbered HU file is missing.
- The loader chooses randomly from the available simulated files, or uses the real file if not found.

### 🧪 Testing

```
<dataroot>/
├── 2539499/
│   └── 43/
│       └── 12_HU.mat
├── 2495118/
│   └── 1/
│       └── 7_HU.mat

```

- Each case contains a single `.mat` file ending with `_HU.mat`.
- The loader uses the first file matching `*_HU.mat` within each subfolder.
- The variable loaded is `sim_hu_int`, or `slice_metal_HU` if not present.

---
## 🧪 Artifact Simulation with MATLAB

If you start from a clean dataset (artifact-free CT slices), you can generate synthetic metal artifacts using the provided MATLAB script.

Every .nii.gz file is expected to be a volume: (x, y, z), where z is the number of slices available.

You can use a .csv file to filter the range of slices to be computed, for instance, to exclude hips.

### 📍 Script Location

```
artifact_simulator/simulator.m
```

### ▶️ How to Use

1. Open `simulator.m` in MATLAB.
2. Edit the paths at the top of the script to match your dataset:
   ```matlab
   folder_GT = "processed_dataset/gt/";        % Ground truth slices (PNG)
   folder_sim_art = "processed_dataset/art/";  % Output folder for artifacts
   folder_masks = "masks\";                    % Binary masks for metal
   ```
3. Make sure you have:
   - A raw dataset in `.nii.gz` format in `raw_dataset/`
   - Metal masks in PNG format under `masks/`

4. Run the script:
   ```matlab
   simulator.m
   ```

### 💡 Output

Each processed slice generates:
- `N_HU.mat` — simulated HU image with artifacts
- `N_mask.mat` — corresponding metal mask
- `N.mat` — artifact image (uint8)
- `gt.mat` — original clean slice (HU)
- PNG thumbnails of both GT and artifacted slices

Organized in subfolders:
```
SpineWeb_paired/dataset/
├── 2495819/
│   ├── 65/
│   │   ├── 1_HU.mat
│   │   ├── 1_mask.mat
│   │   ├── 1.mat
│   │   ├── 2_HU.mat
│   │   └── ...
│   └── gt.mat
└── ...
```

These outputs are directly compatible with the `UnalignedCalimarganDataset` loader for training.

---

## 🧾 Configuration Files

Configuration files are located in the `configs/` directory and use the `.json` format.

### 🔧 Example: `configs/train_config.json`

```json
{
  "dataroot": "path/to/dataset",
  "name": "calimar_gan_exp1",
  "model": "calimar_gan",
  "dataset_mode": "unaligned_calimargan",
  "load_size": 256,
  "crop_size": 256,
  "batch_size": 8,
  "niter": 60,
  "niter_decay": 40,
  "gan_mode": "lsgan",
  "lambda_A": 10,
  "lambda_B": 10,
  "lambda_identity": 0.8,
  "no_dropout": true
}
```

### 🧪 Example: `configs/test_config.json`

```json
{
  "dataroot": "path/to/dataset",
  "name": "calimar_gan_exp1",
  "model": "calimar_gan",
  "dataset_mode": "unaligned_calimargan",
  "epoch": 100,
  "batch_size": 64,
  "load_size": 256,
  "crop_size": 256,
  "no_dropout": true,
  "no_flip": true,
  "saveDisk": true,
  "phase": "test"
}
```

All options from `TrainOptions` and `TestOptions` are supported and can be overridden via these config files.

---

## 📈 Outputs and Checkpoints

- Logs and models are saved in:  
  `checkpoints/<experiment_name>/`

- Training/test options are automatically saved as:  
  `checkpoints/<experiment_name>/train_opt.txt` or `test_opt.txt`

---
## Citations

```
@article{SCARDIGNO2025102565,
title = {CALIMAR-GAN: An unpaired mask-guided attention network for metal artifact reduction in CT scans},
journal = {Computerized Medical Imaging and Graphics},
volume = {123},
pages = {102565},
year = {2025},
issn = {0895-6111},
doi = {https://doi.org/10.1016/j.compmedimag.2025.102565},
url = {https://www.sciencedirect.com/science/article/pii/S0895611125000746},
author = {Roberto Maria Scardigno and Antonio Brunetti and Pietro Maria Marvulli and Raffaele Carli and Mariagrazia Dotoli and Vitoantonio Bevilacqua and Domenico Buongiorno}
}
```
---
## Contact
If you have any question, please feel free to contact Roberto Scardigno (Email: r.scardigno.3@gmail.com)
