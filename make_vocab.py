# @Time    : 2019/4/17 21:43
# @Author  : shakespere
# @FileName: make_vocab.py
from collections import Counter
from data import DPDataSet
from voc import Voc
VOC_FILE = "./data/vocab.csv"
def make_vocab(path):
    voc = Voc()
    train_dataset = DPDataSet("train")
    for _,sentence in train_dataset:
        voc.add_sentence(sentence)
    counter = Counter(voc.gram2count)
    with open(path,"w") as f:
        for word,count in counter.most_common():
            f.write(word + ',' + str(count) + '\n')
    print("Build Vocab Done!")
if __name__ == "__main__":
    make_vocab(VOC_FILE)