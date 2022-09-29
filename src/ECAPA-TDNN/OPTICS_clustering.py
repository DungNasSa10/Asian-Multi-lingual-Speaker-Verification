from sklearn.cluster import OPTICS
from ECAPAModel import ECAPAModel
import torch.nn.functional as F
import tools 
import argparse, glob, os, torch, warnings, time
import numpy as np
import soundfile
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

#Ecapa model 
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')
parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--n_class', type=int,   default=5994,   help='Number of speakers')
parser.add_argument('--save_path',  type=str,   default="",  help='')


#Initialize
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = tools.init_args(args)


clust = OPTICS( metric='cosine', eps=args.eps, min_cluster_size=args.min_cluster_size, n_jobs=args.n_cpu)

s = ECAPAModel(**vars(args))
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
    cluster = clust.fit_predict(embeddings)
    vid_audio.append(audio_list)
    vid_embedding.append(embeddings)
    vid_cluster.append(cluster)
  print(vid_cluster)

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
    # print(i/len(speaker_dict.keys()))
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

with open(args.save_file, 'w') as f:
  for i in range(len(clusters)):
    for j in list(clusters.values())[i]:
     f.write(str(list(clusters.keys())[i])+ '\t' + str(j[1])+'\n')


