# ADP-Net



## Environment Setup

### 1. 3D Image Warping

- **Environment:** MATLAB R2018a or later.

### 2. Inpainting

1. Clone this repository:

   ```bash
   git clone https://github.com/threedteam/ADP-Net.git
   cd ADP-Net
   ```

2. Create and activate a Conda environment (recommended):

   ```bash
   conda create -n adpnet python=3.8
   conda activate adpnet
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```



## Running Process

### Step 1: Train the ADP-Net Model

1. Data Preparation:

   The training data directly uses the Reference View. The reference view and its corresponding binarized instance segmentation masks are cropped into a large number of small image patches to build the training dataset. Please use data/txt.py to prepare the file list for your reference view data.

2. Configure Training Parameters:

   Modify the configs/config_train.yml file to set hyperparameters such as the dataset path.

3. **Start Training:**

   ```bash
   python train.py -c configs/config_train.yml -e your_experiment_name --resume_mae your_mae_pretrain
   ```

   - You can resume training from a checkpoint using the `--resume` argument.

   - Training logs and model checkpoints will be saved in the `ckpts/your_experiment_name` directory.

     

### Step 2: Generate Guidance Masks and Target Views from the Reference View

1. Navigate to the `Masked 3D Image Warping with RDIG` directory.
2. Modify the file paths in the main script, using the **Reference View**, its depth map, and instance segmentation data (with instance IDs) as input.
3. Run the script. This script will perform the 3D warping and **produce two outputs**:
   - The **Target View image with holes and its corresponding hole mask**.
   - The pruned and binarized Instance Guidance Mask.



### Step 3: Perform Inference with ADP-Net

1. Prepare Test Data:

   Ensure that the test data path specified in configs/config_test.yml is correct. This data should be the data generated in Step 2.

2. **Run Inference:**

   Bash

   ```Bash
   python test.py -c configs/config_test.yml -r path/to/your/checkpoint.ckpt --load_pl
   ```

   - `-r` specifies the path to the model checkpoint you trained in Step 1.
   - `--load_pl` indicates loading a PyTorch Lightning format checkpoint.
   - `--output_path` can be used to specify the output directory for the results.



### Step 4: Performance Evaluation

Use the `data/metric.py` script to calculate various metrics for the generated results.

```Bash
python data/metric.py --real_path path/to/ground_truth --fake_path path/to/your_results --info_txt result.txt
```



## Acknowledgments

Our codes are based on [LaMa](https://github.com/advimman/lama), [MAE](https://github.com/facebookresearch/mae) and [LHF](https://github.com/threedteam/dibr).