import sys
from humanmvmt.cnn_new.model import *
from humanmvmt.lrcn_attention.model import *
from humanmvmt.lrcn_attention.data_prep import LRCNDatarandomLoader
from humanmvmt.trainer_new import Trainer
from humanmvmt.data_overlap import Overlap_dataloader_1, Overlap_dataloader_5
from humanmvmt.utils import lrcn_accuracy_score
from humanmvmt.probability import Probability
import numpy as np

num_channel = 4
data_path = '../datas/feature4_40_5/'
#data_path = './data/'
# Instantiate optimal CNN model object
cnn_model = CNN_C(num_channels = num_channel, is_lrcn=True)

def load_fine_tune_model(model, pre_model):
    pre_dict = pre_model.state_dict()
    model_dict = model.state_dict()
    pre_dict = {k: v for k,v in pre_dict.items() if k in model_dict}
    #pre_dict.pop("linear.weight")
    #pre_dict.pop("linear.bias")
    #pre_dict.pop("label.weight")
    #pre_dict.pop("label.bias")
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)
    return model

# 0 CNN_F: original CNN with full connect
# 1 LRCN_F: original LRCN_F
# 4 LRCN_F1: change view to permute
# 2 LRCN_FA: F1 with attention
# 3 LRCN_FA_Probability: FA with probability
mode = ['CNN_G', 'LRCN_F', 'LRCN_FA', 'LRCN_FA_Probability', 'LRCN_F1', 'LSTM_B', 'LSTM_B1']
model_index = int(sys.argv[1])

overlap_types=['no_overlap', 'sample_overlap', 'segment_overlap']
sample_types=['random', 'sequential']
num_overlap = int(sys.argv[2])
num_sample = int(sys.argv[3])
offset = int(sys.argv[4])

if model_index >= len(mode):
    print(f"mode index error\n avaliable mode: {mode}")
    exit()
model_name = mode[ model_index ]
overlap_name = overlap_types[num_overlap]
sampling_name = sample_types[num_sample]
print(f"model name:{model_name}, overlap type:{overlap_types[num_overlap]},\
      sampling type:{sample_types[num_sample]}")
# Load data
#data_path = './data/'
#loader = LRCNDatarandomLoader(data_path, batch_size = 64, train_split = 0.7)
prefix_name = f"{model_name}_{overlap_name}_{sampling_name}"

if model_index >0 and model_index < 5:
    loader = Overlap_dataloader_5(data_path, overlap_types[num_overlap], sample_types[num_sample], batch_size = 64, train_shuffle = True)
    train_loader, test_loader, valid_loader = loader.load()
    #using_pre_model = True
    using_pre_model = False
    if using_pre_model:
        try:
            #model = torch.load(f'Models/{prefix_name}_final', map_location=torch.device('cpu')) 
            #print("Pre trained model load successfully!")
    
            pre_model = torch.load('Models/' + mode[4] + '_best', map_location=torch.device('cpu')) 
            model = locals()[model_name](cnn_model, num_channels=4, segment_size=40, num_classes=5)
            model = load_fine_tune_model(model, pre_model)
            print("Pre trained model load successfully!")
        except:
            model = locals()[model_name](cnn_model, num_channels=num_channel, segment_size=40, num_classes=5)
    else:
        model = locals()[model_name](cnn_model, num_channels=num_channel, segment_size=40, num_classes=5)
    
    if model_name == 'LRCN_FA_Probability':
        prob = Probability(train_loader, saved_prob_path = 'Models/probability.npy', saved_means_path = 'Models/prob_means.pkl')
        prob.set_probability()
        tr = Trainer(model, train_dataloader = train_loader,\
            val_dataloader = valid_loader, prob = prob, prefix_model_name = prefix_name, need_prob=False)
    else:
        tr = Trainer(model, train_dataloader = train_loader,\
            val_dataloader = valid_loader, prefix_model_name = prefix_name, need_prob=False)
else:
    loader = Overlap_dataloader_1(data_path, overlap_types[num_overlap], sample_types[num_sample], batch_size = 64, train_shuffle = True)
    train_loader, test_loader, valid_loader = loader.load()
    prefix_name = f"{model_name}_{overlap_name}_{sampling_name}"
    model = locals()[model_name](num_channels=num_channel, segment_size=40, classes=5)
    tr = Trainer(model, train_dataloader = train_loader,\
        val_dataloader = valid_loader, prefix_model_name = prefix_name, need_prob=False)

    
#tr.fit(batch_size = 64, epochs = 200)

# Get Prediction
y_te_pred, y_te_true, y_te_out, y_orig_outs, raw_maxs = tr.predict(test_loader,\
        saved_model_path = f"./Models/{prefix_name}_best" )
print('Test Accuracy: ', lrcn_accuracy_score(y_te_pred, y_te_true))
file_path = f'/data/grp_pshakari/mvmt/{model_name}_{overlap_name}_{sampling_name}/'                                                                                                                                  
import os                                                                                                                                                                                               
if not os.path.exists(file_path):                                                                                                                                                                       
    os.makedirs(file_path)                                                                                                                                                                              
file_name = f'test_pred.npy'                                                                                                                                                              
with open(file_path + file_name, 'wb') as f:                                                                                                                                                            
    y_te_pred = np.reshape(np.array(y_te_pred), (-1,))
    np.save(f, y_te_pred)  
file_name = f'test_true.npy'                                                                                                                                                              
with open(file_path + file_name, 'wb') as f:                                                                                                                                                            
    y_te_true = np.reshape(np.array(y_te_true), (-1,))
    np.save(f, y_te_true)  
file_name = f'test_out.npy'                                                                                                                                                              
with open(file_path + file_name, 'wb') as f:                                                                                                                                                            
    y_te_out = np.reshape(np.array(y_te_out), (-1,))
    np.save(f, y_te_out)  
