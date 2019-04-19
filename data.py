# @Time    : 2019/4/17 19:32
# @Author  : shakespere
# @FileName: data.py
from os.path import join
from torch.utils.data import Dataset,DataLoader
class DPDataSet(Dataset):
    def __init__(self,split,data_dir="./data/"):
        assert split in ["train","test"]
        self.split = split
        self.data_dir = data_dir
        self.pairs = self.load_data()
    def load_data(self):
        pairs = []
        with open(join(self.data_dir,self.split + ".csv")) as f:
            for line in f:
                label,sentence = line.split(",",1)
                label = label.strip('"')
                sentence = sentence.strip('"').strip('\n')
                label = int(label) - 1
                pairs.append((label,sentence))
        return pairs
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, index):
        return self.pairs[index]
