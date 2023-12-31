# AAD-MovingSpeakers

## Overview
This repository supports our research paper titled "Brain-controlled augmented hearing for spatially moving conversations in noisy environments". The main components of this repository are:
1. **Binaural Speech Separation Algorithm**: Separates the speech streams of the moving talkers while preserving their location.
2. **Auditory Attention Decoding (AAD)**: Decodes to which talker the listener is attending to by analyzing their brain signals.

> 🚨 **Notice**: The research paper has not yet been made public. This code is currently intended for paper review purposes only.

## 1. Separating Moving Speakers in a Sound Mixture
This section provides data, code for training separation models, pre-trained models, and a demo for inference.

### Pre-requisites
- Ensure you have installed all the dependencies listed in the `requirements.txt` file.
- This codebase is tested on Python version 3.9.16.

### Datasets
- The [Google Resonance Audio Software Development Kit](https://resonance-audio.github.io/resonance-audio/) was employed to spatialize the audio. For more details about spatializing sounds through HRTFs, adding reverb, and modeling shoebox environments, please refer to these [scripts](https://github.com/vishalchoudhari11/GoogleResonanceAudioSpatializer).
- We provide [pre-generated moving speaker audio](https://drive.google.com/file/d/1XFhzlkn6UKcSMa4JOJXIqj1RkwFpkPre/view?usp=sharing). You can download them without the need to generate them by yourself.
- Download [DEMAND dataset](https://zenodo.org/record/1227121) for acoustic noise in diverse environments.

### Training 
We train both a separation model, a post-enhancement model, and a localization model separately.

#### Separation Model
- After downloading the pre-generated moving speaker audio and noise audio, set up the dataset:
  ```bash
  python create_separation_dataset.py
- Train the separation model:
  ```bash
  python train_separation_model.py --training-file-path 'your_path' --validation-file-path 'your_path' --checkpoint-path 'your_path'

#### Post-enhancement model
- After training the separation model, please use it to separate speakers and create the dataset for training the enhancement model.
- Kick off the enhancement model training:
  ```bash
  python train_enhancement_model.py --training-file-path 'your_path' --validation-file-path 'your_path' --checkpoint-path 'your_path'

#### Trajectory prediction model
- The localization model is used to predict the locations (moving trajectory) of the separated speaker.
- After training the enhancement model, please use it to get enhanced separated speech and create the dataset for training the localization model.
- Train the localization model training:
  ```bash
  python train_localization_model.py --training-file-path 'your_path' --validation-file-path 'your_path' --checkpoint-path 'your_path'

## 2. Auditory Attention Decoding (AAD)
This section contains resources and code for conducting AAD and relevant analyses.

### Training CCA models:

- The script [Step_15_Spec_SS_g_PCA_CCA_FINAL.m](https://github.com/naplab/AAD-MovingSpeakers/blob/main/AAD/Analysis%20Scripts/Step_15_Spec_SS_g_PCA_CCA_FINAL.m) is used to train CCA models that learn the mapping between the neural responses and the attended stimuli. 

### Evaluating the CCA models:

- The script [Step_15_Spec_SS_WinByWin_PCA_CCA_FINAL.m](https://github.com/naplab/AAD-MovingSpeakers/blob/main/AAD/Analysis%20Scripts/Step_15_Spec_SS_WinByWin_PCA_CCA_FINAL.m) is used to evaluate the CCA models for various window sizes on a window-by-window basis and also generate correlations of the brain waves with the attended and unattended stimuli.

We use the CCA implementation from the [NoiseTools package](http://audition.ens.fr/adc/NoiseTools/) developed by Dr. Alain de Cheveigné:

de Cheveigné, A., Wong, DDE., Di Liberto, GM, Hjortkjaer, J., Slaney M., Lalor, E. (2018) Decoding the auditory brain with canonical correlation analysis. NeuroImage 172, 206-216, [https://doi.org/10.1016/j.neuroimage.2018.01.033](https://doi.org/10.1016/j.neuroimage.2018.01.033).

