import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.model_utils import cLN, TCN


class SeparationNet(nn.Module):
    """Separation network"""
    def __init__(
        self, 
        enc_dim = 64,
        feature_dim = 64,
        hidden_dim = 256, 
        enc_win = 64,
        enc_stride = 32,
        num_block = 5,
        num_layer = 7,
        kernel_size = 3,
        stft_win = 512,
        stft_hop = 32,
        num_spk = 2,
    ):
        super(SeparationNet, self).__init__()
    
        # hyper parameters
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = self.feature_dim*4

        self.enc_win = enc_win
        self.enc_stride = enc_stride

        self.num_layer = num_layer
        self.num_block = num_block
        self.kernel_size = kernel_size

        self.stft_win = stft_win
        self.stft_hop = stft_hop
        self.stft_dim = stft_win//2 + 1

        self.num_spk = num_spk

        # input encoder
        self.encoder1 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        self.encoder2 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        # Causal layer normalization
        self.LN = cLN(self.enc_dim*2, eps=1e-8)
        # Bottom neck layer
        self.BN = nn.Conv1d(self.enc_dim*2+self.stft_dim*3, self.feature_dim, 1, bias=False)

        # TCN encoder
        self.TCN = TCN(self.feature_dim, self.enc_dim*self.num_spk*2, self.num_layer, 
                       self.num_block, self.hidden_dim, self.kernel_size, causal=True)
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.enc_win, stride=self.enc_stride, bias=False)

        self.eps = 1e-12


    def padding(self, feature, window):

        batch_size, n_ch, n_sample = feature.shape

        rest = window - (self.enc_stride + n_sample % window) % window
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, n_ch, rest)).type(feature.type())
            feature = torch.cat([feature, pad], 2)

        if window == 512:
            pad_aux = Variable(torch.zeros(batch_size, n_ch, self.enc_stride+120)).type(feature.type())
        else:
            pad_aux = Variable(torch.zeros(batch_size, n_ch, self.enc_stride)).type(feature.type())
            
        feature = torch.cat([pad_aux, feature, pad_aux], 2)
    
        return feature, rest


    def get_frequency_features(self, waveform):

        input_fd, _ = self.padding(waveform, self.stft_win)

        phase_left = torch.stft(input_fd[:,0,:], self.stft_win, hop_length=self.stft_hop, window=torch.hann_window(self.stft_win).to(waveform.device), center=False, return_complex=True)
        phase_right = torch.stft(input_fd[:,1,:], self.stft_win, hop_length=self.stft_hop, window=torch.hann_window(self.stft_win).to(waveform.device), center=False, return_complex=True)

        phase_left = torch.view_as_real(phase_left)
        phase_right = torch.view_as_real(phase_right)

        # IPD and IID
        IPD = torch.atan2(phase_left[:,:,:,1],phase_left[:,:,:,0]) - torch.atan2(phase_right[:,:,:,1],phase_right[:,:,:,0])
        IPD_cos = torch.cos(IPD)
        IPD_sin = torch.sin(IPD)
        IPD_feature = torch.cat([IPD_cos, IPD_sin], 1)

        IID = torch.log(phase_left[:,:,:,1]**2 + phase_left[:,:,:,0]**2 + self.eps) - torch.log(phase_right[:,:,:,1]**2 + phase_right[:,:,:,0]**2 + self.eps)

        freq_feature = torch.cat([IPD_feature, IID], 1)

        return freq_feature


    def forward(self, input):
        """
        Args: 
            input: mixed waveform (B, 2, L)

        Returns:
            separated waveforms in the left channel (B, num_spk, L)
            separated waveforms in the right channel (B, num_spk, L)

        """

        batch_size, n_ch, n_sample = input.shape # B, 2, L

        input_td, rest = self.padding(input, self.enc_win) 

        # encoder
        enc_map_left = self.encoder1(input_td[:,0,:].unsqueeze(1))  # B, N, T
        enc_map_right = self.encoder2(input_td[:,1,:].unsqueeze(1))  # B, N, T

        # Apply layer normalization
        enc_features = self.LN(torch.cat([enc_map_left, enc_map_right], dim=1))

        # Cross domain features
        freq_features = self.get_frequency_features(input)
        freq_features = freq_features[:, :, :enc_features.shape[-1]]
        all_features = torch.cat([enc_features, freq_features], 1)

        # TCN separator
        # generate C feature matrices for the C speakers
        mask = torch.sigmoid(self.TCN(self.BN(all_features))).view(batch_size, self.enc_dim, self.num_spk, 2, -1)  # B, H, 2, T

        # left channel
        output_left = torch.cat([mask[:, :, i, 0, :] * enc_map_left for i in range(self.num_spk)], 0)  # B*num_spk, H, T
        output_left = self.decoder(output_left)  # B*C, 1, L
        output_left = output_left[:,:,self.enc_stride:-(rest+self.enc_stride)].contiguous()  # B*num_spk, 1, L
        output_left = torch.cat([output_left[batch_size*i:batch_size*(i+1), :, :] for i in range(self.num_spk)], 1)  # B, num_spk, L

        # right channel
        output_right = torch.cat([mask[:, :, i, 1, :] * enc_map_right for i in range(self.num_spk)], 0)  # B*num_spk, H, T
        output_right = self.decoder(output_right)  # B*C, 1, L
        output_right = output_right[:,:,self.enc_stride:-(rest+self.enc_stride)].contiguous()  # B*num_spk, 1, L
        output_right = torch.cat([output_right[batch_size*i:batch_size*(i+1), :, :] for i in range(self.num_spk)], 1)  # B, num_spk, L

        return output_left, output_right
    
    
class EnhancementNet(nn.Module):
    """Post enhancement module"""
    def __init__(
        self, 
        enc_dim = 64,
        feature_dim = 64,
        hidden_dim = 256, 
        enc_win = 64,
        enc_stride = 32,
        num_block = 5,
        num_layer = 7,
        kernel_size = 3,
        num_spk = 1,
    ):
        super(EnhancementNet, self).__init__()
    
        # hyper parameters
        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        self.hidden_dim = self.feature_dim*4

        self.enc_win = enc_win
        self.enc_stride = enc_stride

        self.num_layer = num_layer
        self.num_block = num_block
        self.kernel_size = kernel_size

        self.num_spk = num_spk

        # input encoder
        self.encoder1 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        self.encoder2 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        self.encoder3 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        self.encoder4 = nn.Conv1d(1, self.enc_dim, self.enc_win, stride=self.enc_stride, bias=False)
        # Causal layer normalization
        self.LN = cLN(self.enc_dim*4, eps=1e-8)
        # Bottom neck layer
        self.BN = nn.Conv1d(self.enc_dim*4, self.feature_dim, 1, bias=False)

        # TCN encoder
        self.TCN = TCN(self.feature_dim, self.enc_dim*4, self.num_layer, 
                       self.num_block, self.hidden_dim, self.kernel_size, causal=True)
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.enc_win, stride=self.enc_stride, bias=False)

        self.eps = 1e-12


    def padding(self, feature, window):

        batch_size, n_ch, n_sample = feature.shape

        rest = window - (self.enc_stride + n_sample % window) % window
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, n_ch, rest)).type(feature.type())
            feature = torch.cat([feature, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, n_ch, self.enc_stride)).type(feature.type())
            
        feature = torch.cat([pad_aux, feature, pad_aux], 2)
    
        return feature, rest

    
    def forward(self, mixture, separated_waveform):
        """
        Args: 
            mixture: mixed waveform (B, 2, L)
            separated_waveform: separated waveform of the target speaker from the separation module (B, 2, L)

        Returns:
            separated waveforms in the left channel (B, num_spk, L)
            separated waveforms in the right channel (B, num_spk, L)

        """
        batch_size, n_ch, n_sample = mixture.shape # B, 2, L

        mixture_td, rest = self.padding(mixture, self.enc_win) 
        separated_waveform_td, _ = self.padding(separated_waveform, self.enc_win) 

        # encoder
        mix_enc_map_left = self.encoder1(mixture_td[:,0,:].unsqueeze(1))  # B, N, T
        mix_enc_map_right = self.encoder2(mixture_td[:,1,:].unsqueeze(1))  # B, N, T
        sep_enc_map_left = self.encoder3(separated_waveform_td[:,0,:].unsqueeze(1))  # B, N, T
        sep_enc_map_right = self.encoder4(separated_waveform_td[:,1,:].unsqueeze(1))  # B, N, T

        # Apply layer normalization
        enc_features = self.LN(torch.cat([mix_enc_map_left, mix_enc_map_right, sep_enc_map_left, sep_enc_map_right], dim=1))

        # TCN separator
        # generate C feature matrices for the C speakers
        masks = torch.sigmoid(self.TCN(self.BN(enc_features))).view(batch_size, self.enc_dim, 2, 2, -1)  # B, H, 2, T
        # mask and sum
        masked_feature_left =  masks[:,:,0,0,:] * mix_enc_map_left + masks[:,:,0,1,:] * mix_enc_map_right
        masked_feature_right = masks[:,:,1,0,:] * mix_enc_map_right + masks[:,:,1,1,:] * mix_enc_map_left
        masked_features = torch.cat([masked_feature_left, masked_feature_right], dim=0)  # 2*B, H, T
        
        # waveform decodersws
        output = self.decoder(masked_features)  # 2*B, 1, L
        output = output[:,:,self.enc_stride:-(rest+self.enc_stride)].contiguous()  # 2*B, 1, L
        output = torch.cat([output[batch_size*i:batch_size*(i+1), :, :] for i in range(2)], 1)  # B, 2, L
        
        return output