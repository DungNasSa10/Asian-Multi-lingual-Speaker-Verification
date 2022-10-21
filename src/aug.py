import numpy
import pandas as pd
import random
from scipy import signal
from scipy.io.wavfile import write
import soundfile, librosa
import glob, os
from tqdm import tqdm


def data_statistic(language_metadata_path: str):

    abbrev2lan = {"en": "English", "fr": "French", "hi": "Hindi", "ja": "Japanese", "ta": "Tamil", "th": "Thai", "uz": "Uzbekistan", "vi": "Vietnameese", "zh-CN": "Chinese"}
    languages_stat = [] # list of languages stat

    with open(language_metadata_path) as f_read:
        language_stat_dict = {} # dict of (speaker_id, language) as key and [wav_paths] as value
        language_stat_list = [] # list of (speaker_id, language, #utterance, [wav_paths])
        lines = f_read.readlines()
        for line in lines:
            speaker_id, wav_path = line.strip().split("\t")
            language_abbrev = wav_path.split('/')[0]
            language = abbrev2lan[language_abbrev]
            if language_stat_dict.get((speaker_id, language)) == None:
                language_stat_dict[(speaker_id, language)] = [wav_path]
            else:
                language_stat_dict[(speaker_id, language)].append(wav_path)
        for speaker_id, language in language_stat_dict.keys():
            wav_paths = language_stat_dict[(speaker_id, language)]
            language_stat_list.append((speaker_id, language, len(wav_paths), wav_paths))
    languages_stat.extend(language_stat_list)
    
    languages_stat = sorted(languages_stat, key=lambda x: (x[1], x[2], x[0]))
    stat_df = pd.DataFrame(languages_stat, columns=["Speaker ID", "Language", "#Utterances", "Wav paths"])

    return stat_df

def loadWAV(filename, max_audio=0, max_frames=0, evalmode=True, num_eval=10):

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]
    if max_audio != 0:
        max_audio = max_audio
    else:
        max_audio = audiosize

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0).astype(numpy.float64)

    return feat


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):

        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15], 'music':[5,15]}
        self.numnoise   = {'noise':[1,10], 'music':[1,10] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []
        self.max_audio = audio.shape[1]
        # print(max_audio)
        for noise in noiselist:
            noiseaudio  = loadWAV(noise, max_audio=self.max_audio, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        
        add_noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return add_noise/4 + audio

    def reverberate(self, audio):
        max_audio = audio.shape[1]
        rir_file    = random.choice(self.rir_files)
        rir, sr     = librosa.load(rir_file, sr=None, mono=True)
        rir         = numpy.expand_dims(rir.astype(numpy.float64),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:max_audio]


class Get_aug_data():
    def __init__(self, train_list, musan_path, rir_path,train_path, max_frames=900,  **kwargs):
        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)

        self.train_list = train_list
        self.max_frames = max_frames
        self.musan_path = musan_path
        self.rir_path   = rir_path

        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_list  = []
        self.data_label = []
        
        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            filename = os.path.join(train_path,data[1])
            
            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def augment(self, x):
        feat = []
                
        audio = loadWAV(x, evalmode=False)

        augtype = random.randint(1,3)
        if augtype == 1:
            audio   = self.augment_wav.reverberate(audio)
        elif augtype == 2:
            audio   = self.augment_wav.additive_noise('music',audio)
        elif augtype == 3:
            audio   = self.augment_wav.additive_noise('noise',audio)
        
        feat.append(audio)
        feat = numpy.concatenate(feat, axis=0)

        return feat[0]
                    

if __name__ == '__main__':
    # Change language here
    # {"en": "English", "fr": "French", "hi": "Hindi", "ja": "Japanese", "ta": "Tamil", "th": "Thai", "uz": "Uzbekistan", "vi": "Vietnameese", "zh-CN": "Chinese"}
    language = 'Vietnameese'
    file_aug = 'vi'

    metadata_path = 'data/MSV_CommonVoice_data/metadata/all_metadata.txt'
    data_stat = data_statistic(metadata_path)
    data_stat = data_stat[data_stat['Language']==language]

    train_list = metadata_path
    musan_path = "data/musan_augment"
    rir_path = "data/rirs_noises"
    train_path = "data/MSV_CommonVoice_data/unzip_data/"

    getter = Get_aug_data(train_list, musan_path, rir_path,train_path)
    min_utter = 6

    with open(f'data/MSV_CommonVoice_data/metadata/aug_{file_aug}.txt', 'w') as f:
        for index, row in tqdm(data_stat.iterrows()):
            if row['#Utterances'] < min_utter:
                for j in range(min_utter - len(row['Wav paths'])):
                    path_name = random.sample(row['Wav paths'], 1)[0]
                    audio = getter.augment(train_path+path_name)
                    write("data/aug2/"+path_name[:-4]+f"({j+1})"+path_name[-4:], 16000, audio)
                    f.write(row['Speaker ID']+'\t'+path_name[:-4]+f"({j+1})"+path_name[-4:]+'\n')
        