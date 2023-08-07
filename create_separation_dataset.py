import os
import pickle
from random import shuffle
import numpy as np
import glob
import h5py
import librosa

np.random.seed(seed=20230215)

train_spk1_path = './train_spat_segment1'
train_spk2_path = './train_spat_segment2'

train_pair_path = './train_pair.pkl'

# Two moving speakers are more closer to each other
train_pair_closer_path = './train_pair_closer.pkl'
# trajactory for training dataset
train_trajactory_path = './traj_segment_train.npy'


with open(train_pair_path,"rb") as f:
    train_pair = pickle.load(f)
    
with open(train_pair_closer_path,"rb") as f:
    train_pair = pickle.load(f)
    
with open(train_trajactory_path,'rb') as f:
    train_traj = np.load(f)
    
    
eval_spk1_path = './eval_spat_segment1'
eval_spk2_path = './eval_spat_segment2'

val_pair_path = './val_pair.pkl'
eval_pair_path = './eval_pair.pkl'

eval_trajactory_path = './traj_segment_eval.npy'

with open(val_pair_path,"rb") as f:
    val_pair = pickle.load(f)
    
with open(eval_pair_path,"rb") as f:
    eval_pair = pickle.load(f)
    
with open(eval_trajactory_path,'rb') as f:
    eval_traj = np.load(f)
    
    
sampling_rate = 16000
noise_path = './demand/'
kitchen,_ = librosa.load(noise_path+'DKITCHEN/ch01.wav', sr=sampling_rate)
meeting,_ = librosa.load(noise_path+'OMEETING/ch01.wav', sr=sampling_rate)
cafe,_ = librosa.load(noise_path+'PCAFETER/ch01.wav', sr=sampling_rate)
rest,_ = librosa.load(noise_path+'PRESTO/ch01.wav', sr=sampling_rate)
subway,_ = librosa.load(noise_path+'PSTATION/ch01.wav', sr=sampling_rate)
car,_ = librosa.load(noise_path+'TBUS/ch01.wav', sr=sampling_rate)
metro,_ = librosa.load(noise_path+'TMETRO/ch01.wav', sr=sampling_rate)
traffic,_ = librosa.load(noise_path+'STRAFFIC/ch01.wav', sr=sampling_rate)
living,_ = librosa.load(noise_path+'DLIVING/ch01.wav', sr=sampling_rate)
washing,_ = librosa.load(noise_path+'DWASHING/ch01.wav', sr=sampling_rate)
field,_ = librosa.load(noise_path+'NFIELD/ch01.wav', sr=sampling_rate)
park,_ = librosa.load(noise_path+'NPARK/ch01.wav', sr=sampling_rate)
river, _ = librosa.load(noise_path+'NRIVER/ch01.wav', sr=sampling_rate)
hall, _ = librosa.load(noise_path+'OHALLWAY/ch01.wav', sr=sampling_rate)
office, _ = librosa.load(noise_path+'OOFFICE/ch01.wav', sr=sampling_rate)
scafe, _ = librosa.load(noise_path+'SCAFE/ch01.wav', sr=sampling_rate)

print([len(kitchen),
        len(meeting),
        len(cafe),
        len(rest),
        len(subway),
        len(car),
        len(metro),
        len(traffic),
        len(living),
        len(washing),
        len(field),
        len(park),
        len(river),
        len(hall),
        len(office),
        len(scafe)])

min_len = np.min([len(kitchen),
                  len(meeting),
                  len(cafe),
                  len(rest),
                  len(subway),
                  len(car),
                  len(metro),
                  len(traffic),
                  len(living),
                  len(washing),
                  len(field),
                  len(park),
                  len(river),
                  len(hall),
                  len(office),
                  len(scafe)])

mean = 0
std = 1 
num_samples = min_len
white = np.random.normal(mean, std, size=num_samples)

all_noise = [kitchen,
             meeting,
             cafe,
             rest,
             subway,
             car,
             metro,
             traffic, living, washing, field, park, river, hall, office, scafe,
             white]
    
    
def create_train_dataset(save_path):
    
    sr = 16000
    utt_len = 9.6
    actual_len = int(utt_len*sr)
    
    dset = h5py.File(save_path, 'w')
    spk1_set = dset.create_dataset('spk1', shape=(24000,2,actual_len), dtype=np.float32)
    spk2_set = dset.create_dataset('spk2', shape=(24000,2,actual_len), dtype=np.float32)
    trace_set = dset.create_dataset('trace', shape=(24000,2,96), dtype=np.int)
    nos_set = dset.create_dataset('noise', shape=(24000,1,actual_len), dtype=np.float32)
    
    for index in range(24000):
        if index < 12000:
            index_pair = train_pair[index]
        else:
            index_pair = train_pair_closer[index-12000]
        
        # load pair of moving speakers
        spk1_spatial, e_sr = librosa.load(os.path.join(train_spk1_path, f"train_b2_{index_pair[0]}.wav"), sr=16000, mono=False)       
        spk2_spatial, e_sr = librosa.load(os.path.join(train_spk2_path, f"train_b2_{index_pair[1]}.wav"), sr=16000, mono=False) 
        # normalize the speech power
        spk1_spatial = spk1_spatial / np.sqrt(np.sum(spk1_spatial[0]**2+spk1_spatial[1]**2)+1e-8) * 1e2
        spk2_spatial = spk2_spatial / np.sqrt(np.sum(spk2_spatial[0]**2+spk2_spatial[1]**2)+1e-8) * 1e2
        # randomly set the snr between two speakers
        spk_snr = np.random.uniform(low=0.0, high=5.0, size=1)[0]
        spk2_spatial = spk2_spatial * np.power(10, spk_snr/20.)
        
        # randomly choose a noise
        noise_type = np.random.randint(low=0, high=17,size=1)[0]
        noise_index = np.random.randint(30*16000, min_len-30*16000, 1)[0]
        noise_snr = np.random.uniform(-15, 2.5, 1)[0]
        noise = all_noise[noise_type][noise_index:noise_index+actual_len]
        noise = noise / np.sqrt(np.sum(noise**2)+1e-8) * np.sqrt(np.sum((spk1_spatial[0]+spk2_spatial[0])**2)+1e-8)
        noise = noise * np.power(10, noise_snr/20.)

        # normalize the waveform
        mixture_spatial = spk1_spatial + spk2_spatial + noise
        scaler = 0.9/np.max(np.abs(mixture_spatial))
        mixture_spatial = mixture_spatial*scaler
        spk1_spatial = spk1_spatial*scaler
        spk2_spatial = spk2_spatial*scaler
        noise = noise*scaler
        
        spk1_set[index, :,:] = spk1_spatial
        spk2_set[index, :,:] = spk2_spatial
        trace_set[index, 0,:] = np.round(train_traj[index_pair[0],0,:])
        trace_set[index, 1,:] = np.round(train_traj[index_pair[1],1,:])
        nos_set[index, :, :] = noise

        
def create_val_and_eval_dataset(save_path, spk_pairs):
    
    sr = 16000
    utt_len = 9.6
    actual_len = int(utt_len*sr)
    
    dset = h5py.File(save_path, 'w')
    spk1_set = dset.create_dataset('spk1', shape=(len(spk_pairs),2,actual_len), dtype=np.float32)
    spk2_set = dset.create_dataset('spk2', shape=(len(spk_pairs),2,actual_len), dtype=np.float32)
    trace_set = dset.create_dataset('trace', shape=(len(spk_pairs),2,96), dtype=np.int)
    nos_set = dset.create_dataset('noise', shape=(len(spk_pairs),1,actual_len), dtype=np.float32)
    
    for index in range(len(spk_pairs)):
        
        index_pair = spk_pairs[index]
        # load pair of moving speakers
        spk1_spatial, e_sr = librosa.load(os.path.join(eval_spk1_path, f"eval_b2_{index_pair[0]}.wav"), sr=16000, mono=False)       
        spk2_spatial, e_sr = librosa.load(os.path.join(eval_spk2_path, f"eval_b2_{index_pair[1]}.wav"), sr=16000, mono=False) 
        # normalize the speech power
        spk1_spatial = spk1_spatial / np.sqrt(np.sum(spk1_spatial[0]**2+spk1_spatial[1]**2)+1e-8) * 1e2
        spk2_spatial = spk2_spatial / np.sqrt(np.sum(spk2_spatial[0]**2+spk2_spatial[1]**2)+1e-8) * 1e2
        # randomly set the snr between two speakers
        spk_snr = np.random.uniform(low=0.0, high=5.0, size=1)[0]
        spk2_spatial = spk2_spatial * np.power(10, spk_snr/20.)
        
        # randomly choose a noise
        noise_type = np.random.randint(low=0, high=17,size=1)[0]
        noise_index = np.random.randint(30*16000, min_len-30*16000, 1)[0]
        noise_snr = np.random.uniform(-15, 2.5, 1)[0]
        noise = all_noise[noise_type][noise_index:noise_index+actual_len]
        noise = noise / np.sqrt(np.sum(noise**2)+1e-8) * np.sqrt(np.sum((spk1_spatial[0]+spk2_spatial[0])**2)+1e-8)
        noise = noise * np.power(10, noise_snr/20.)

        # normalize the waveform
        mixture_spatial = spk1_spatial + spk2_spatial + noise
        scaler = 0.9/np.max(np.abs(mixture_spatial))
        mixture_spatial = mixture_spatial*scaler
        spk1_spatial = spk1_spatial*scaler
        spk2_spatial = spk2_spatial*scaler
        noise = noise*scaler
        
        spk1_set[index, :,:] = spk1_spatial
        spk2_set[index, :,:] = spk2_spatial
        trace_set[index, 0,:] = np.round(eval_traj[index_pair[0],0,:])
        trace_set[index, 1,:] = np.round(eval_traj[index_pair[1],1,:])
        nos_set[index, :, :] = noise

        
def mean():
    create_train_dataset('your train save path') 
    create_val_and_eval_dataset('your val save path', val_pair)
    create_val_and_eval_dataset('your eval save path', eval_pair)
          
if __name__ == '__main__':
    main()