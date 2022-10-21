
from speechbrain.processing.PLDA_LDA import *
import pickle
import random, numpy
import torch.nn.functional as F
import sys, os, argparse
import yaml
import numpy
import torch
from utils import *
from SpeakerNet import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser(description = "PLDA")

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')


parser.add_argument('--public_test_list',   type=str,   default='')
parser.add_argument('--public_test_path',   type=str,   default='')

parser.add_argument('--train_embedding_file',   type=str,   default='/home/tuht/Asian-Multi-lingual-Speaker-Verification/temporary/---metadata/plda_test/train_emb.pickle')
parser.add_argument('--enroltest_embedding_file',   type=str,   default='/home/tuht/Asian-Multi-lingual-Speaker-Verification/temporary/---metadata/plda_test/enroltest_emb.pickle')
parser.add_argument('--train_stat_file',   type=str,   default='/home/tuht/Asian-Multi-lingual-Speaker-Verification/temporary/---metadata/plda_test/train_emb.pickle')
parser.add_argument('--enrol_stat_file',   type=str,   default='/home/tuht/Asian-Multi-lingual-Speaker-Verification/temporary/---metadata/plda_test/train_emb.pickle')
parser.add_argument('--test_stat_file',   type=str,   default='/home/tuht/Asian-Multi-lingual-Speaker-Verification/temporary/---metadata/plda_test/train_emb.pickle')
parser.add_argument('--ndx_file',   type=str,   default='/home/tuht/Asian-Multi-lingual-Speaker-Verification/temporary/---metadata/plda_test/train_emb.pickle')

parser.add_argument('--train_list1',   type=str,   default='/home/tuht/Asian-Multi-lingual-Speaker-Verification/temporary/---metadata/plda_test/train_emb.pickle')
parser.add_argument('--train_path1',   type=str,   default='/home/tuht/Asian-Multi-lingual-Speaker-Verification/temporary/---metadata/plda_test/enroltest_emb.pickle')
   
parser.add_argument('--gpu',     type=int,   default=0)
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread', type=int, default=4,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

parser.add_argument('--n_mels',         type=int,   default=80,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder')
parser.add_argument('--sinc_stride',    type=int,   default=10,     help='Stride size of the first analytic filterbank layer of RawNet3')
parser.add_argument('--C',              type=int,   default=1024,   help='Channel size for the speaker encoder (ECAPA_TDNN)')

parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')
parser.add_argument('--nClasses',       type=int,   default=17714,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')
parser.add_argument('--lr',             type=float, default=0.0005,  help='Learning rate')
parser.add_argument('--weight_decay',   type=float, default=2e-5,      help='Weight decay in the optimizer')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [lr_step] epochs')
parser.add_argument('--lr_step',        type=int,   default=2,      help='Step for learning rate decay')

parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_file',      type=str,   default="exps/exp1", help='Path for model and logs')

args = parser.parse_args()

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

def create_embedding_dict(name, data_list, data_path, model, embedding_dict_file):
    model.eval()
    if not os.path.isfile(embedding_dict_file):
        file_filter = {}
        talkative_speaker = []
        files = []
        embedding_dict = {} 
        lines = open(data_list).read().splitlines()
        for line in tqdm.tqdm(lines, total = len(lines)):
            if name == 'test':
                files.append(line.split()[0])
                files.append(line.split()[1])
            else:
                if line.split()[0] not in  list(file_filter.keys()):
                    file_filter[line.split()[0]] = [line.split()[1]]
                    files.append(line.split()[1])
                elif len(file_filter[line.split()[0]]) > args.max_seg_per_spk :
                    file_filter[line.split()[0]] += [line.split()[1]]
                    talkative_speaker.append(line.split()[0])
                else:
                    file_filter[line.split()[0]] += [line.split()[1]]
                    files.append(line.split()[1])
        for i in tqdm.tqdm(list(set(talkative_speaker)), total = len(list(set(talkative_speaker)))):
            file_list = random.sample(list(file_filter[i]),args.max_seg_per_spk-1)
            files += file_list
        setfiles = list(set(files))
        setfiles.sort()
        
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _  = soundfile.read(os.path.join(data_path, file))
            # Maximum audio length
            max_audio = args.max_frames * 160 + 240

            audiosize = audio.shape[0]

            if audiosize <= max_audio:
                shortage    = max_audio - audiosize + 1 
                audio       = numpy.pad(audio, (0, shortage), 'wrap')
                audiosize   = audio.shape[0]
            startframe = numpy.linspace(0,audiosize-max_audio, num=10)
            feats = []
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            if name == 'train':
                audio = numpy.stack(feats,axis=0).astype(numpy.float32)                
                data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()
                with torch.no_grad():
                    embedding = model(data_1)
                    embedding = F.normalize(embedding, p=2, dim=1)
                embedding_dict[file] = embedding
            else:
                embeddings = []
                with torch.no_grad():
                   for i in feats:
                       embedding = model(torch.FloatTensor(i).cuda())
                       embedding = F.normalize(embedding, p=2, dim=1)
                       embeddings.append(embedding)
                embedding_dict[file] = embeddings
        
        with open(embedding_dict_file, "wb") as input:
            pickle.dump(embedding_dict, input, pickle.HIGHEST_PROTOCOL) 
    else:
        with open(embedding_dict_file, "rb") as input:
            embedding_dict = pickle.load(input)

    return embedding_dict

def emb_computation_loop(split,data_list,embedding_dict, stat_file):
    """Computes the embeddings and saves the in a stat file"""
    # Extract embeddings (skip if already done)
    if not os.path.isfile(stat_file):
        if split == 'train':
            embeddings =[]
            modelset = []
            segset = []

            for i in range(len(data_list)):
                if data_list[i][1] in embedding_dict.keys():
                
                    modelset.append(data_list[i][0])
                    emb = embedding_dict[data_list[i][1]]
                    segset.append(split+str(i))
                    embeddings.append(emb.squeeze().cpu().numpy())
                
            embeddings = F.normalize(torch.FloatTensor(numpy.stack(embeddings)), p=2, dim=1).squeeze(1).cpu().numpy() #numpy.sum(embeddings, axis = 1)/10
            modelset = numpy.array(modelset, dtype="|O")
            segset = numpy.array(segset, dtype="|O")

            # Intialize variables for start, stop and stat0
            s = numpy.array([None] * embeddings.shape[0])
            b = numpy.array([[1.0]] * embeddings.shape[0])

            # Stat object (used to collect embeddings)
            stat_obj = StatObject_SB(
            modelset=modelset,
            segset=segset,
            start=s,
            stop=s,
            stat0=b,
            stat1=embeddings,
            )
            stat_obj.save_stat_object(stat_file)
        else:
            stat_obj_list = []
            for i in range(len(data_list)):
               modelset = []
               segset = []
               for j in range(len(embedding_dict[data_list[i]])):
                   modelset.append(split+str(i)+ str(j))
                   segset.append(split+str(i)+str(j))
               embeddings = [i.squeeze().cpu().numpy() for i in embedding_dict[data_list[i]]]
               embeddings = numpy.stack(embeddings)
               modelset = numpy.array(modelset, dtype="|O")
               segset = numpy.array(segset, dtype="|O")
               s = numpy.array([None] * embeddings.shape[0])
               b = numpy.array([[1.0]] * embeddings.shape[0])
               stat_obj = StatObject_SB(
                    modelset=modelset,
                    segset=segset,
                    start=s,
                    stop=s,
                    stat0=b,
                    stat1=embeddings,
               )
               stat_obj_list.append(stat_obj)
            with open(stat_file, "wb") as output:
                pickle.dump(stat_obj_list, output, pickle.HIGHEST_PROTOCOL)
            return stat_obj_list

    else:
        with open(stat_file, "rb") as input:
            stat_obj = pickle.load(input)

    return stat_obj

s= SpeakerNet(**vars(args))
s = WrappedModel(s).cuda(args.gpu)
trainer     = ModelTrainer(s, **vars(args))
trainer.loadParameters(args.initial_model)
model =  trainer.__model__

train_data = pd.read_csv(args.train_list1, sep = '\t', header =None)
data_list = train_data.values.tolist()
train_embedding_dict = create_embedding_dict('train',args.train_list1, args.train_path1, model, args.train_embedding_file)
plda = PLDA( rank_f = 100, nb_iter = 10, scaling_factor = 0.05)
train_obj = emb_computation_loop("train",data_list, train_embedding_dict, args.train_stat_file)
plda.plda(train_obj)

enroltest_data = pd.read_csv(args.public_test_list, sep = ' ', header =None)  
enrol_list =  enroltest_data[enroltest_data.columns[0]].values.tolist()
test_list = enroltest_data[enroltest_data.columns[1]].values.tolist()
enroltest_embedding_dict = create_embedding_dict('test',args.public_test_list, args.public_test_path, model, args.enroltest_embedding_file)

enrol_obj = emb_computation_loop("enrol", enrol_list,enroltest_embedding_dict, args.enrol_stat_file)
print(type(enrol_obj))
print(type(enrol_obj[1]))
test_obj = emb_computation_loop("test",test_list, enroltest_embedding_dict, args.test_stat_file)
   
if not os.path.isfile(args.ndx_file):
    ndx_obj_list = []
    for i in range(len(enrol_list)):
      models = enrol_obj[i].modelset
      testsegs = test_obj[i].modelset

      ndx_obj = Ndx(models=models, testsegs=testsegs)
      ndx_obj_list.append(ndx_obj)
    with open(args.ndx_file, "wb") as output:
       pickle.dump(ndx_obj_list, output, pickle.HIGHEST_PROTOCOL)
else:
    with open(args.ndx_file, "rb") as input:
            ndx_obj_list = pickle.load(input)
scores = [] 
for i in tqdm.tqdm(range(len(enrol_list)), total = len(enrol_list)):
  scores_plda = fast_PLDA_scoring(enrol_obj[i], test_obj[i], ndx_obj_list[i], plda.mean, plda.F, plda.Sigma)
  scores.append(scores_plda.scoremat.mean())
# scores_plda = fast_PLDA_scoring(enrol_obj, test_obj, ndx_obj, plda.mean, plda.F, plda.Sigma)
scaler = MinMaxScaler(feature_range=(0, 1),clip = True)
# final_scores = scaler.fit_transform(scores_plda.scoremat.diagonal().reshape(-1,1))
final_scores = scaler.fit_transform(np.array(scores).reshape(-1,1))
with open(args.save_file, 'w') as handle:
    k = 0
    for i in final_scores:
        handle.write(str(enrol_list[k])+'\t'+str(test_list[k])+'\t'+str(i[0])+'\n')
        k += 1
