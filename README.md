# AAD-MovingSpeakers

This repository accompanies our research paper titled "Brain-controlled augmented hearing for spatially moving conversations in noisy environments". It contains the dataset and codes we use for training the separation models, the pre-trained models and the inference demo, and the code we use to perform the auditory attention decoding study. 

*Notice: The paper has not been publicly available. This code is intended for paper review only.*

## Pre-requisites
Install everything listed in the `requirements.txt` file. A note on the Python version: we tested our environment on Python 3.9.16. 

## Datasets
1. [Google Resonance Audio software development kit](https://resonance-audio.github.io/resonance-audio/) was used to spatialize the audio streams of the conversations. Please also check the [scripts](https://github.com/vishalchoudhari11/GoogleResonanceAudioSpatializer) that allow one to spatialize sounds through HRTFs, add reverb, and model shoebox environments.
2. We provide pre-generated raw audio. Please download them from here.

## Training 
We train a separation model, post-enhancement, separately. 

### Separation model
After downloading the pre-generated raw audio, create the dataset for training the separation model:
```bash
python create_datasts.py
```
Then, start training the separation model:
```bash
python train_separation_model.py --training-file-path 'your_path' --validation-file-path 'your_path' --checkpoint-path 'your_path'
```
### Post-enhancement model
After training the separation model, please use it to separate speakers and create the dataset for training the enhancement model.

Then, start training the enhancement model:
```bash
python train_enhancement_model.py --training-file-path 'your_path' --validation-file-path 'your_path' --checkpoint-path 'your_path'
```





