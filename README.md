# AAD-MovingSpeakers

## Overview
This repository supports our research paper titled "Brain-controlled augmented hearing for spatially moving conversations in noisy environments". The main components of this repository are:
1. **Binaural speech separation algorithm**: Separates the speech streams of the talkers while preserving their location.
2. **Auditory attention decoding (AAD)**: Identifies whom the listener wants to listen to by reading brain signals.

> 🚨 **Notice**: The research paper has not yet been made public. This code is currently intended for paper review purposes only.

## Separating Moving Speakers in a Sound Mixture
This section provides data, code for training separation models, pre-trained models, and a demo for inference.

### Pre-requisites
- Ensure you have installed all the dependencies listed in the `requirements.txt` file.
- This codebase is tested on Python version 3.9.16.

### Datasets
1. The [Google Resonance Audio software development kit](https://resonance-audio.github.io/resonance-audio/) was employed to spatialize the audio streams of the conversations. For spatializing sounds through HRTFs, adding reverb, and modeling shoebox environments, refer to these [scripts](https://github.com/vishalchoudhari11/GoogleResonanceAudioSpatializer).
2. Pre-generated moving speaker audio is available. You can download them here.
3. Download [DEMAND dataset](https://zenodo.org/record/1227121) for acoustic noise in diverse environments.

### Training 
We train both a separation model and a post-enhancement model, separately.

#### Separation Model
- After downloading the pre-generated raw audio, set up the dataset:
  ```bash
  python create_datasets.py
- Train the separation model:
  ```bash
  python train_separation_model.py --training-file-path 'your_path' --validation-file-path 'your_path' --checkpoint-path 'your_path'

#### Post-enhancement model
- After training the separation model, please use it to separate speakers and create the dataset for training the enhancement model.
- Kick off the enhancement model training:
  ```bash
  python train_enhancement_model.py --training-file-path 'your_path' --validation-file-path 'your_path' --checkpoint-path 'your_path'

## Audotry Attention Decoding (AAD)
This section contains resources and code for conducting AAD and relevant analyses.





