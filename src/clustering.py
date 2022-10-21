from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
from SpeakerNet import *
import torch.nn.functional as F
import yaml 
import argparse, glob, os, torch, warnings
import numpy as np
import soundfile
import random
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description = "OPTICS_clustering")
#Argument setting
parser.add_argument('--youtube_path',  type=str,   default="",  help='Path of youtube data')
parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')
parser.add_argument('--save_file',  type=str,   default="",  help='Path of save file for new labeled data')
parser.add_argument('--min_cluster_size', type=int,   default=3,     help='min point of a cluster ')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--gpu',      type=int,   default=0,       help='GPU')
parser.add_argument('--thres',      type=float,   default=0.7,       help='threshold for 1 people in 2 vid')
parser.add_argument('--max_eps',         type=float, default=0.35,   help='max distance for a cluster')
parser.add_argument('--initial_model',  type=str,   default="",  help='Path of the initial_model')
parser.add_argument('--all',    dest='all', action='store_true', help='All the file in youtube or cluster in each vid ')
parser.add_argument('--n_components', type=int,   default=3,     help='number of cluster for gmm ')
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


#Initialize
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
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


clust = OPTICS( metric='cosine', max_eps=args.max_eps, min_cluster_size=args.min_cluster_size, n_jobs=args.n_cpu)
clust2 = GaussianMixture(n_components = args.n_components)

n = SpeakerNet(**vars(args))
n = WrappedModel(n).cuda()
s = ModelTrainer(n, **vars(args))
s.loadParameters(args.initial_model)
s.__model__.eval()
if args.all:
  embeddings = []
  youtube_list_path = glob.glob(args.youtube_path+'/*/*.wav')
  for i, audio_path in tqdm.tqdm(enumerate(youtube_list_path), total = len(youtube_list_path)):
    audio, _ = soundfile.read(audio_path)
    audio = torch.FloatTensor(np.stack([audio], axis=0)).cuda()
    embedding = s.__model__(audio)
    embedding = F.normalize(embedding, p=2, dim=1).squeeze(0).detach().cpu().numpy()
    embeddings.append(embedding)
  cluster = clust.fit_predict(embeddings)
  print(cluster)
  clusters={}
  for index in range(len(youtube_list_path)):
    if cluster[index] != -1:
      if cluster[index] not in clusters.keys():
        clusters[cluster[index]] = [(embeddings[index],youtube_list_path[index][len(args.youtube_path)+1:])]
      else:
        clusters[cluster[index]] += [(embeddings[index],youtube_list_path[index][len(args.youtube_path)+1:])]


else:
  print('still in developing process')
  youtube_list_path = os.listdir(args.youtube_path) 

  vid_audio = []
  vid_embedding = []
  vid_cluster = []
  for i, vid in tqdm.tqdm(enumerate(youtube_list_path), total = len(youtube_list_path)):
    embeddings = []
    audio_list = [str(vid)+'/'+str(i) for i in os.listdir(args.youtube_path+'/'+vid)]
    for audio_path in audio_list:
      audio,_ = soundfile.read(args.youtube_path+'/'+audio_path)
      audio = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
      embedding = s.__model__(audio)#.squeeze(0).detach().cpu().numpy()
      embedding = F.normalize(embedding, p=2, dim=1).squeeze(0).detach().cpu().numpy()
      embeddings.append(embedding)
    try:
      cluster = clust2.fit_predict(embeddings)
    except Exception:
      print(vid)
      continue
    vid_audio.append(audio_list)
    vid_embedding.append(embeddings)
    vid_cluster.append(cluster)

  speaker_dict={}
  for index in range(len(vid_audio)):
    for i in range(len(vid_cluster[index])):
      if vid_cluster[index][i] != -1:
        if str(index)+'-'+str(vid_cluster[index][i]) not in speaker_dict.keys():
          speaker_dict[str(index)+'-'+str(vid_cluster[index][i])] = [(vid_embedding[index][i],vid_audio[index][i])]
        else:
          speaker_dict[str(index)+'-'+str(vid_cluster[index][i])] += [(vid_embedding[index][i],vid_audio[index][i])]
    

  # for i in range(len(speaker_dict)):
  #   print(str(list(speaker_dict.keys())[i])+ '\t' + str([i[1] for i in list(speaker_dict.values())[i]]))

  checking_ind = list(range(len(speaker_dict.keys())))
  clusters = {}
  for i in range(len(speaker_dict.keys())):
    if i >= len(speaker_dict.keys()):
      break
    # print(i/len(speaker_dict.keys()))
    # Invalid Speaker Removal
    invalid_sm = cosine_similarity([speaker_embed[0] for speaker_embed in list(speaker_dict.values())[i]], \
                             [speaker_embed[0] for speaker_embed in list(speaker_dict.values())[i]])
    if np.min(invalid_sm) < 0.5:
      speaker_dict.pop(list(speaker_dict.keys())[i])


  for i in range(len(speaker_dict.keys())):
    # print(i/len(speaker_dict.keys()))
    # Invalid Speaker Removal
    if i not in checking_ind:
      continue
    clusters[list(speaker_dict.keys())[i]] = speaker_dict[list(speaker_dict.keys())[i]]
    for j in range(i+1, len(speaker_dict.keys())):
      if j not in checking_ind:
        continue
      score = cosine_similarity([speaker_embed[0] for speaker_embed in list(speaker_dict.values())[i]], \
                             [speaker_embed[0] for speaker_embed in list(speaker_dict.values())[j]])
      if score.mean() >= args.thres:
        checking_ind.remove(j)
        clusters[list(speaker_dict.keys())[i]] += speaker_dict[list(speaker_dict.keys())[j]]

shuffle_list = []
with open(args.save_file, 'w') as f:
  for i in range(len(clusters)):
    for j in list(clusters.values())[i]:
      shuffle_list.append((list(clusters.keys())[i],j[1]))
  random.shuffle(shuffle_list)
  for i in range(len(shuffle_list)):
    f.write(str(shuffle_list[i][0])+ '\t' + str(shuffle_list[i][1])+'\n')