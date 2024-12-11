# Dark-Matter-Diffusion

The cosmic web, shaped by the distribution of dark matter, defines the large-scale structure of the universe. Galaxies serve as biased tracers of this web, but their distributions are influenced by both dark matter clustering and other astrophysical processes, introducing uncertainties in cosmological reconstructions. Prior work has used diffusion to reconstruct dark matter fields from stellar mass distributions, marginalizing over astrophysical and cosmological uncertainties. We build a diffusion model capable of conditioning on additional data types, including stellar mass maps, fast radio burst (FRB) maps and gravitational lensing maps, to refine dark matter reconstructions. FRBs provide dispersion measures that probe ionized gas, while gravitational lensing directly traces dark matter through light deflection, with lensing shear maps derived from total mass distributions via a Fast Fourier Transform (FFT). These multi-modal inputs improve the model’s ability to disentangle astrophysical uncertainties and enhance accuracy. Our multi-modal model progressively denoises noisy dark matter maps, achieving cross-correlation values exceeding 0.75 across CAMELS datasets. We evaluate the impact of varying signal-to-noise ratio (SNR) for each modality, demonstrating improved reconstruction fidelity with higher SNR values. Finally, applying the model to Hubble Space Telescope shear maps conditioned on dark matter and Astrid lensing data validates its real-world applicability, advancing our understanding of the cosmic web and establishing a robust framework for future dark matter studies. 

<!-- ![Alt text](/assets/sample.png) -->

## Setup on HPC

For computing, we used both Google Colab and Caltech HPC. We show here how to set up the codebase on HPC (Caltech's High Performance Computing clusters).

To SSH into HPC:

```console
$ ssh username@hpc.caltech.edu
[username@login1 ~]$ cd /central/groups/mlprojects/dm_diff/Dark-Matter-Diffusion/
[username@login1 Dark-Matter-Diffusion]$ ls
```

You need your access password and Duo 2FA authentication (for Caltech HPC).

### Conda environment

To install conda:
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ./Miniconda3-latest-Linux-x86_64.sh
```
To set up conda using new installation:
```sh
source ~/.bashrc
```
To set up to use jupyter notebook (not strictly required):
```sh
conda install ipykernel
python -m ipykernel install --user --name base --display-name "Python (base)"
conda install -c anaconda jupyter
```

To create our environment, run:

```console
$ conda env create -f environment.yml -n dm_diff
$ conda init bash
$ conda init zsh
$ source ~/.bashrc
```

To activate the environment, run the following.
```console
(base) [username@login1 Dark-Matter-Diffusion]$ conda activate dm_diff
(dm_diff) [username@login1 Dark-Matter-Diffusion]$ 
```

You will need to activate the conda environment every time you connect via SSH. You can avoid this by doing:
```console
(dm_diff) [username@login1 Dark-Matter-Diffusion]$ echo "conda activate dm_diff" >> ~/.bashrc
```

### Compute on HPC
Most of our code requires access to computing resources. To run the code on HPC, either batch a job with the below commands, or use the following command to create an interactive gpu-enabled terminal:
```console
srun --pty -A <account-to-charge> -t <days>-<hours>:<mins>:<seconds> -n 1 -N 1 --gres=gpu:1 -p gpu /bin/bash -l
```
Where `-n` is the number of tasks to run, `-N` is the number of nodes requested. We are also requesting a gpu partition with one gpu, and telling it to open up a login shell on this machine.

## Data
We use data from the CAMELS Multifield Dataset. Data can be accessed at [this url](https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/). We use the following data for each modality, with option to choose between Astrid, SIMBA, and IllustrisTNG simulation datasets. Our models are currently set up for 2D data (in the 2D directory of the above url).

|Modality       |Filename                              |
|---------------|--------------------------------------|
|Dark Matter    | `Maps_Mcdm_{dataset}_LH_z=0.00.npy`  |
|Stellar Mass   | `Maps_Mstar_{dataset}_LH_z=0.00.npy` |
|Gas Mass       | `Maps_Mgas_{dataset}_LH_z=0.00.npy`  |
|Mtot           | `Maps_Mtot_{dataset}_LH_z=0.00.npy`  |

Note: Lensing maps are generated from the `Mtot` maps prior to training, and fast radio burst maps are generated from `Mgas` maps during training.

If running on Caltech hpc, these data can also be accessed at the following path: `/groups/mlprojects/dm_diffusion/data/`

### Generating Lensing Data
The lensing maps are generated from the `Mtot` maps prior to training. This pipeline is set up to work on the Caltech hpc. If running here, just input the dataset as an argument, else you will need to make a few minor changes to `src/gravitational_lensing_gen.py`.

To generate lensing data, run
```console
python src/gravitational_lensing_gen.py --dataset <dataset>
```
Where `<dataset>` is replaced by 'Astrid', 'IllustrisTNG', or 'SIMBA'.

Below are visuals of what these guides should look like.
![Alt text](/assets/baseline_guides.png)

## Train Debiasing Diffusion Models

After setup, you can run the following to train:
```console
python src/train.py --exp_name default_train
```

This will create a folder called `model_out/<exp_name>/<timestamp>/`. In training, the model is saved at each epoch in a file called `model_epoch_<epoch>.pt`, the config file for this model is saved in `config.yaml`, and the losses are saved in `losses.npy`. 

The following files and visualizations are saved in `model_out/<exp_name>/<timestamp>/ep<epoch>/`  at each epoch:
| Filename             | Description                                   |
|----------------------|-----------------------------------------------|
| final_sample.png     | Image of the final epoch sample.              |
| gt.png               | Image of the ground truth.                    |
| gt.pt                | Raw ground truth.                             |
| guide_frb.png        | Guide fast radio burst image.                 |
| guide_frb.pt         | Raw guide fast radio burst.                   |
| guide_lens1.png      | Guide lensing channel 1.                      |
| guide_lens2.png      | Guide lensing channel 2.                      |
| guide_lensing1.pt    | Raw guide lensing 1.                          |
| guide_lensing2.pt    | Raw guide lensing 2.                          |
| guide_stellar.png    | Guide stellar image.                          |
| guide_stellar.pt     | Raw guide stellar image.                      |
| sample.png           | Sampling progression.                         |
| sample999.pt         | Sample at final timestep (default 1000 steps).|

The following arguments may be adjusted in terminal for training:

| Argument               | Default     | Help                                              |
|------------------------|-------------|---------------------------------------------------|
| --dataset              | Astrid      | IllustrisTNG, Astrid, or SIMBA.                   |
| --epochs               | 30          | Number of epochs for training.                    |
| --batch_size           | 12          | Batch size for training.                          |
| --timesteps            | 1000        | Timesteps for training.                           |
| --learning_rate        | 0.0001      | Learning rate for training.                       |
| --img_size             | 256         | Image size. Single int, (H = W).                  |
| --unconditional        | False       | Enable unconditional mode (flag to enable).       |
| --debug                | False       | Enable debug mode (flag to enable).               |
| --out_path             | /model_out/ | Path to save models.                              |
| --sigma_noise_lensing  | 0           | Noise for lensing noise experiment.               |
| --sigma_noise_stellar  | 0           | Noise for mstar noise experiment.                 |
| --perc_preserved_frb   | 10          | Percent mgas preserved for fast radio burst calc. |
| --no_stellar           | False       | Disable Stellar (include flag to omit).           |
| --no_frb               | False       | Disable FRB (include flag to omit).               |
| --no_lensing           | False       | Disable Lensing (include flag to omit).           |
| --exp_name             | None        | Experiment name (required).                       |


### Model Checkpoints
We have uploaded model checkpoints corresponding to the experiments presented in our paper to `/checkpoints/<model_type>/`. Each checkpoint directory comes with a `config.yaml` file. Directories in this file will need to be updated based on where the data is located. Our baseline model (most corrupted input data to simulate real measurements) is located in `checkpoints/stellar0.1_lens10_FRB1p_256/`.

## Sampling from Trained Models
![Alt text](/assets/baseline_mean_std.png)

To sample from a created model, run the following command:
```console
python src/sample_multiple.py --model_path <path_to_model>
```
Replacing `<path_to_model>` with the `.pt` file corresponding to your model. 

Upon sampling, the following directory will be created in the epoch folder corresponding to your model:
`model_out/<exp_name>/<timestamp>/ep<epoch>/sampling_<timestamp of sampling>/`.

The following flags are used to run `sample_multiple.py`:
| Argument     | Default | Required | Help                                         |
|--------------|---------|----------|----------------------------------------------|
| --model_path | None    | Yes      | Path to specific model.                      |
| --idx        | None    | No       | Specific index in dataset of sample.         |
| --N          | 10      | No       | Number of samples to generate.               |


The following files will be created in this folder after sampling. Additionally, each sample will be saved as `sample_<i>.pt`.
| Filename            | Description                                           |
|---------------------|-------------------------------------------------------|
| all_samples.png     | grid of individual samples                            |
| corr.png            | power spectra correlations                            |
| dm_correlation.png  | correlation box plot - gt vs target (dm maps)         |
| frb_guide.png       | fast radio burst guide image                          |
| frb_guide.pt        | fast radio burst guide raw                            |
| ground_truth.png    | ground truth image                                    |
| ground_truth.pt     | ground truth raw                                      |
| lens1_guide.png     | lens ch1 guide image                                  |
| lens1_guide.pt      | lens ch1 guide raw                                    |
| lens2_guide.png     | lens ch2 guide image                                  |
| lens2_guide.pt      | lens ch2 guide raw                                    |
| mean_std.png        | plot of ground truth, mean, and std of samples        |
| metrics.pt          | file with psnr, mse, corr vals                        |
| power_spectra.png   | power spectra plot - gt vs target (dm maps)           |
| stellar_guide.png   | stellar mass guide image                              |
| stellar_guide.pt    | stellar mass guide raw                                |

To reproduce our experiments and compare metrics across several models, use `sample_comparison.py`. This will compare several runs and produce the above plots, comparing metrics where applicable.
The following flags are used to run this.

| Argument          | Default | Required | Help                                                  |
|-------------------|---------|----------|-------------------------------------------------------|
| --baseline_path   | None    | Yes      | Path to baseline model.                               |
| --mod1_path       | None    | Yes      | Path to mod1 model.                                   |
| --mod2_path       | None    | Yes      | Path to mod2 model.                                   |
| --change          | None    | Yes      | Variable changed: stellar, frb, or lens.              |
| --idx             | None    | No       | Specific index in dataset of sample.                  |


To sample out of distribution (on a new dataset or using fewer modalities), use `sample_multiple_ood.py` with the following arguments. Note, the inputted model must have already been trained with any modality you wish to condition on.

Example run:
```console
python sample_multiple_ood.py --model_path <path-to-model> --stellar --dataset Astrid --N 10
```
Conducts inference, guiding only on the stellar modality. The model passed is trained on stellar, frb, and lensing data.

| Argument      | Default    | Required | Help                                            |
|---------------|------------|----------|-------------------------------------------------|
| --dataset     | 'Astrid'   | No       | Path to specific model.                         |
| --model_path  | None       | Yes      | Path to specific model.                         |
| --batch_size  | 12         | No       | Batch size for training.                        |
| --idx         | None       | No       | Specific index in dataset of sample.            |
| --N           | 10         | No       | Number of samples to generate.                  |
| --stellar     | False      | No       | To condition on stellar (flag).                 |
| --frb         | False      | No       | To condition on frb (flag).                     |
| --lensing     | False      | No       | To condition on lensing (flag).                 |
| --out_path    | sample_ood/| No       | Path to save samples in.                        |


### Metrics/ Visuals
The files `sample_multiple.py` and `sample_comparison.py` calculate our metrics and generate experimental visualizations. 
`sample_multiple.py` generates `N` images from different seeds given the same ground truth (with an option to specify the index of the ground truth) and computes PSNR, MSE, Correlations, Power Spectra Correlations, and Power Spectra over these samples. See the above section for how to run these files.

## Jupyter Notebooks

The notebooks included in this repository are meant to be run in Google Colab. To use, download from this repo, upload into google drive, and select the option to open in Google Colab.