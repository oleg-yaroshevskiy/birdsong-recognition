import librosa
import os
from pydub import AudioSegment

class BirdDataset:
    def __init__(self, df, config, valid=False):
        
        self.filename = df.filename.values
        self.ebird_label = df.ebird_label.values
        self.ebird_code = df.ebird_code.values
        self.sample_rate = config.sample_rate
        
        if valid:
            self.aug = valid_audio_augmentation
        else:
            self.aug = train_audio_augmentation
        
    
    def __len__(self):
        return len(self.filename)
    
    def load_audio(self, path):
        try:
            sound = AudioSegment.from_mp3(path)
            sound = sound.set_frame_rate(self.sample_rate)
            sound_array = np.array(sound.get_array_of_samples(), dtype=np.float32)
        except:
            print("my bad")
            #sound_array = np.zeros(self.sample_rate * args.max_duration, dtype=np.float32)
            
        return sound_array, args.sample_rate

    def __getitem__(self, item):
        
        filename = self.filename[item]
        ebird_code = self.ebird_code[item]
        ebird_label = self.ebird_label[item]

        data = self.load_audio(f"{args.ROOT_PATH}/{ebird_code}/{filename}")
        spect = self.aug(data=data)["data"]
        
        target = ebird_label
        
        return {
            "spect" : torch.tensor(spect, dtype=torch.float), 
            "target" : torch.tensor(target, dtype=torch.long)
        }

class TrainDataset:
    def __init__(self, path, df, sr=32000):
        self.path = path
        self.df = df
        self.sr
        
    def _read_audio_librosa(self, item):
        audio, sr = librosa.load(
            os.path.join(self.path, item.ebird_code, item.filename),
            sr=self.sr)
        return audio, sr
    
    def _read_audio_pydub(self, item):
        try:
            sound = AudioSegment.from_mp3(
                os.path.join(self.path, item.ebird_code, item.filename))
            sound = sound.set_frame_rate(self.sr)
            sound_array = np.array(sound.get_array_of_samples(), dtype=np.float32)
        except:
            print("my bad", item)
            sound_array = np.zeros(self.sr * item.duration, dtype=np.float32)
        
        return sound_array, item.sampling_rate
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        item = self.df.iloc[index]
        audio, sr = self._read_audio_librosa(item)
        
        return audio