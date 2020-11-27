import numpy as np
from python_speech_features import sigproc
import scipy.io.wavfile as wav

class VAD():
    def __init__(self):
        # the vad default parameters, as same as kaldi
        self.frame_length = 0.025
        self.frame_shift = 0.01
        self.sample_rate = 16000
        self.n_fft = 512
        self.vad_energy_threshold = 5.5
        self.vad_energy_mean_scale = 0.5
        self.vad_frames_context = 5
        self.vad_proportion_threshold = 0.6
    
    def get_audio_data(self, audio_path):
        sr, data = wav.read(audio_path)
        assert sr == self.sample_rate
        return data

    def get_speech_features_energy(self, data):
        '''
        :param data: one block data
        :return:
        '''
        signal = sigproc.preemphasis(data, coeff=0.97)
        frames = sigproc.framesig(signal, self.frame_length * self.sample_rate, self.frame_shift * self.sample_rate)
        pspec = sigproc.powspec(frames, self.n_fft)
        energy = np.sum(pspec, 1)  # this stores the total energy in each frame
        energy = np.where(energy == 0, np.finfo(float).eps, energy)
        log_energy = np.log(energy)
        return log_energy
    
    def gen_vad(self, audio_path):
        '''
        :param sequence log_energys
        :return:
        '''
        # get data 
        data = self.get_audio_data(audio_path)
        log_energy = self.get_speech_features_energy(data=data)
        T = len(log_energy)
        energy_threshold = self.vad_energy_threshold + self.vad_energy_mean_scale * sum(log_energy) / T
        output_vad_ft = np.zeros(T)
        for t in range(T):
            log_energy_data = log_energy
            num_count = 0
            den_count = 0
            context = self.vad_frames_context
            for t2 in range(t - context, t + context + 1):
                if t2 >= 0 and t2 < T:
                    den_count += 1
                    if log_energy_data[t] > energy_threshold:
                        num_count += 1
            if num_count >= den_count * self.vad_proportion_threshold:
                output_vad_ft[t] = 1
            else:
                output_vad_ft[t] = 0
        return output_vad_ft

if __name__ == '__main__': 
    wav_path = '../resources/output1.wav'
    vad = VAD()
    vad_fts = vad.gen_vad(wav_path)
    print(vad_fts[:100])