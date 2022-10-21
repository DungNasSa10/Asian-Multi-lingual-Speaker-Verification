import pandas as pd
import yaml, pickle
import argparse, warnings
from sklearn.metrics.pairwise import cosine_similarity as cs_sklearn
from SpeakerNet import *


parser = argparse.ArgumentParser(description = "Asnorm")
# Argument setting
parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file')
parser.add_argument('--save_temporary_path',  type=str,   default="",  help='Path of save file for new labeled data')
parser.add_argument('--embedding_file_path',  type=str,   default="",  help='Path of save file for new embedding data')
parser.add_argument('--embedding_full_file_path',  type=str,   default="",  help='Path of save file for new embedding data')

# Train and test data
parser.add_argument('--train_list',     type=str,   default="data/MSV_CommonVoice_data/metadata/all_new_metadata2.txt",  help='Train list')
parser.add_argument('--test_list',      type=str,   default="data/Test/veri_test2.txt",   help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="data/MSV_CommonVoice_data/", help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="data/Test/wav", help='Absolute path to the test set')

# Runing config
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--gpu',      type=int,   default=0,       help='GPU')
parser.add_argument('--initial_model',  type=str,   default="",  help='Path of the initial_model')
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function')
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler')


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

def create_embedding_dict(name, data_list, data_path, model, embedding_dict_file):
    max_seg_per_spk=200
    model.eval()
    max_frames = 300
    if not os.path.isfile(embedding_dict_file):
    # if True
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
                elif len(file_filter[line.split()[0]]) > max_seg_per_spk:
                    file_filter[line.split()[0]] += [line.split()[1]]
                    talkative_speaker.append(line.split()[0])
                else:
                    file_filter[line.split()[0]] += [line.split()[1]]
                    files.append(line.split()[1])
        for i in tqdm.tqdm(list(set(talkative_speaker)), total = len(list(set(talkative_speaker)))):
            file_list = random.sample(list(file_filter[i]),max_seg_per_spk-1)
            files += file_list
        setfiles = list(set(files))
        setfiles.sort()
        
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _  = soundfile.read(os.path.join(data_path, file))       
            data_1 = torch.FloatTensor(numpy.stack([audio],axis=0))
            with torch.no_grad():
                embedding = model(data_1)
                embedding = F.normalize(embedding, p=2, dim=1)
            embedding_dict[file] = embedding
        
        with open(embedding_dict_file, "wb") as input:
            pickle.dump(embedding_dict, input) 
    else:
        with open(embedding_dict_file, "rb") as input:
            embedding_dict = pickle.load(input)

    return embedding_dict

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

def cosine_score(model, test_list, test_path, full=False, norm=True, embeddings_file='data/embedding_data/embeddings_file.pickle'):
    model.__model__.eval()
    files = []
    embeddings = {}

    cs = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    if "score_norm" in params:
        train_cohort = torch.stack(list(train_embedding_dict.values()))
        
    lines = open(test_list).read().splitlines()
    for line in lines:
        files.append(line.split()[1])
        files.append(line.split()[2])
    setfiles = list(set(files))
    setfiles.sort()

    if not os.path.isfile(embeddings_file):
        if not full:
            print('Loading cut audio')
            for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
                audio, _  = soundfile.read(os.path.join(test_path, file))  
                
                # Splited utterance matrix
                max_audio = 300 * 160 + 240

                if audio.shape[0] <= max_audio:
                    shortage = max_audio - audio.shape[0]
                    audio = numpy.pad(audio, (0, shortage), 'wrap')
                feats = []
                startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
                for asf in startframe:
                    feats.append(audio[int(asf):int(asf)+max_audio])
                feats = numpy.stack(feats, axis = 0).astype(numpy.float)
                data_2 = torch.FloatTensor(feats).cuda()    

                with torch.no_grad():
                    embedding_2 = model.__model__(data_2)
                    embedding_2 = F.normalize(embedding_2, p=2, dim=1)
                embeddings[file] = embedding_2
        else:
            print('loading full audio')
            for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
                audio, _  = soundfile.read(os.path.join(test_path, file)) 
            # Full utterance
                data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()
                with torch.no_grad():
                    embedding_2 = model.__model__(data_1)
                    embedding_2 = F.normalize(embedding_2, p=2, dim=1)
                embeddings[file] = embedding_2


        with open(embeddings_file, "wb") as input:
            pickle.dump(embeddings, input) 
    else:
        with open(embeddings_file, "rb") as input:
            embeddings = pickle.load(input)

    scores  = []

    if norm:
        for line in tqdm.tqdm(lines, total = len(lines)):	
            enrols_embed_12 = embeddings[line.split()[1]]
            
            tests_embed_22= embeddings[line.split()[2]]
            
            norm_score = []

            # Short embedding score
            for i in range(len(enrols_embed_12)):
                try:
                    enrol = torch.FloatTensor(enrols_embed_12[i]).reshape([1,192]).cuda()	
                    test = torch.FloatTensor(tests_embed_22[i]).reshape([1,192]).cuda()
                except:
                    enrol = (enrols_embed_12[i]).reshape([1,192]).cuda()	
                    test = (tests_embed_22[i]).reshape([1,192]).cuda()

                if "score_norm" in params:
                    # Getting norm stats for enrol impostors
                    enrol_rep = enrol.repeat(train_cohort.shape[0], 1, 1)
                    score_e_c = cs(enrol_rep, train_cohort)

                    if "cohort_size" in params:
                        score_e_c = torch.topk(
                            score_e_c, k=params["cohort_size"], dim=0
                        )[0]

                    mean_e_c = torch.mean(score_e_c, dim=0)
                    std_e_c = torch.std(score_e_c, dim=0)

                    # Getting norm stats for test impostors
                    test_rep = test.repeat(train_cohort.shape[0], 1, 1)
                    score_t_c = cs(test_rep, train_cohort)

                    if "cohort_size" in params:
                        score_t_c = torch.topk(score_t_c, k=params["cohort_size"], dim=0)[0] 

                    mean_t_c = torch.mean(score_t_c, dim=0)
                    std_t_c = torch.std(score_t_c, dim=0)

                # Compute the score for the given sentence
                score = cs(enrol, test)[0]

                # Perform score normalization
                if "score_norm" in params:
                    if params["score_norm"] == "z-norm":
                        score = (score - mean_e_c) / std_e_c
                    elif params["score_norm"] == "t-norm":
                        score = (score - mean_t_c) / std_t_c
                    elif params["score_norm"] == "s-norm":
                        score_e = (score - mean_e_c) / std_e_c
                        score_t = (score - mean_t_c) / std_t_c
                        score = 0.5 * (score_e + score_t)
                norm_score.append(score.item())

            scores.append(np.array(norm_score).mean())
    else:
        for line in tqdm.tqdm(lines, total = len(lines)):	
            enrols = embeddings[line.split()[1]]
            tests = embeddings[line.split()[2]]
            score_2 = cs_sklearn(enrols, tests).mean()
            scores.append(score_2)

    return scores


if __name__ == '__main__':
    h = pd.read_csv(args.test_list, sep='\t', header = None)
    h[2] = 0
    h = h[h.columns[[-1,0,1]]]
    h.to_csv(args.save_temporary_path, sep='\t',index=False,header=False)

    n = SpeakerNet(**vars(args))
    n = WrappedModel(n).cuda()
    s = ModelTrainer(n, **vars(args))
    s.loadParameters(args.initial_model)
    model =  s.__model__

    fmt = "\n=== {:30} ===\n"
    train_list1 = args.train_list
    train_path1 = args.train_path
    train_embedding_file = 'data/embedding_data/train_emb_dict.pickle'

    print(fmt.format('Loading train_embedding_dict'))
    train_embedding_dict = create_embedding_dict('train',train_list1, train_path1, model, train_embedding_file)
    print(fmt.format('Loading done'))


    params={
        "score_norm": "s-norm", 
        "cohort_size": 1000
    }

    print(fmt.format('Calculating Score'))
    scores = cosine_score(
        s, 
        args.save_temporary_path, 
        args.test_path,     
        embeddings_file=args.embedding_file_path
    )

    scores_2 = cosine_score(
        s, 
        args.save_temporary_path, 
        args.test_path,     
        full=True,
        embeddings_file=args.embedding_full_file_path
    )

    print(fmt.format('Normalizing Score'))
    scaled = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    scaled_2 = (scores_2 - np.min(scores_2)) / (np.max(scores_2) - np.min(scores_2))
    print(fmt.format('Calculating done!'))

    final_scale = (np.array(scaled) + np.array(scaled_2))/2

    h = pd.read_csv(args.test_list, sep='\t', header = None)
    h[2] = pd.Series(final_scale)
    h.to_csv(args.save_temporary_path, sep='\t',index=False,header=False)