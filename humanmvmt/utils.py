import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)
    

def lrcn_accuracy_score(true_sequence, pred_sequence):
    cnt = 0
    ratio = 1
    for act_seq, pred_seq in zip(true_sequence, pred_sequence):
        if type(act_seq) == torch.Tensor:
            act_seq = act_seq.cpu().detach().numpy()
        try:
            ratio = len(act_seq)
            for act_seg, pred_seg in zip(act_seq, pred_seq):
                if act_seg == pred_seg:
                    cnt+=1
        except:
            if act_seq == pred_seq:
                cnt += 1
    return cnt/(len(pred_sequence)*ratio)
