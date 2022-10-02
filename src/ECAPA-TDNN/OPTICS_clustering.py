from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture
from ECAPAModel import ECAPAModel
import torch.nn.functional as F
import tools 
import argparse, glob, os, torch, warnings, time
import numpy as np
import soundfile
import random
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description = "OPTICS_clustering")
#Argument setting
parser.add_argument('--youtube_path',  type=str,   default="",  help='Path of youtube data')
parser.add_argument('--save_file',  type=str,   default="",  help='Path of save file for new labeled data')
parser.add_argument('--min_cluster_size', type=int,   default=3,     help='min point of a cluster ')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--thres',      type=float,   default=0.7,       help='threshold for 1 people in 2 vid')
parser.add_argument('--eps',         type=float, default=0.2,   help='max distance for a cluster')
parser.add_argument('--initial_model',  type=str,   default="",  help='Path of the initial_model')
parser.add_argument('--all',    dest='all', action='store_true', help='All the file in youtube or cluster in each vid ')
parser.add_argument('--n_components', type=int,   default=3,     help='number of cluster for gmm ')

parser.add_argument('--save_path',  type=str,   default="",  help='')


#Initialize
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = tools.init_args(args)


clust = OPTICS( metric='cosine', eps=args.eps, min_cluster_size=args.min_cluster_size, n_jobs=args.n_cpu)
clust2 = GaussianMixture(n_components = args.n_components)

s = ECAPAModel( 0.001, 0.97, 1024 , 2345, 0.2, 30, 1)
s.load_parameters(args.initial_model)
s.eval()
if args.all:
  embeddings = []
  youtube_list_path = glob.glob(args.youtube_path+'/*/*.wav')
  for audio_path in youtube_list_path:
    audio,_ = soundfile.read(audio_path)
    audio = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
    embedding = s.speaker_encoder.forward(audio, aug = False).squeeze(0).detach().cpu().numpy()
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
  for vid in youtube_list_path:
    embeddings = []
    audio_list = [str(vid)+'/'+str(i) for i in os.listdir(args.youtube_path+'/'+vid)]
    for audio_path in audio_list:
      audio,_ = soundfile.read(args.youtube_path+'/'+audio_path)
      audio = torch.FloatTensor(np.stack([audio],axis=0)).cuda()
      embedding = s.speaker_encoder.forward(audio, aug = False).squeeze(0).detach().cpu().numpy()
      # embedding = F.normalize(embedding, p=2, dim=1).squeeze(0).detach().cpu().numpy()
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
    if i == len(speaker_dict.keys()):
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
