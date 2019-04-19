# @Time    : 2019/4/18 10:41
# @Author  : shakespere
# @FileName: sklearn_main.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import numpy as np
from data import DPDataSet
from utils import prepropress_for_ml
def main():
    """处理数据"""
    train_dataset = DPDataSet('train')
    test_dataset = DPDataSet('test')
    train_labels,train_sents = zip(*train_dataset.pairs)
    test_labels,test_sents = zip(*test_dataset.pairs)
    """将句子分词，因为sklearn使用空格来判断词语之间的界限的"""
    train_sents = prepropress_for_ml(train_sents)
    test_sents = prepropress_for_ml(test_sents)
    """转换向量的形式，使用tf-idf为特征
        处理中文的时候要指定token_pattern参数，因为sklearn中默认丢弃长度为1的token
    """
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    train_sents_tfidf = tfidf.fit_transform(train_sents)
    test_sents_tfidf = tfidf.transform(test_sents)

    """数据准备好后开始训练"""
    lr_clf = LogisticRegression(solver="lbfgs",max_iter=3000)
    lr_clf.fit(train_sents_tfidf,train_labels)
    predicted = lr_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted==np.array(test_labels))
    print("ACC LR:{:.2f}%".format(acc*100))
    """朴素贝叶斯"""
    nb_clf = MultinomialNB()
    nb_clf.fit(train_sents_tfidf,train_labels)
    predicted = nb_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted==np.array(test_labels))
    print("ACC Naive Bayes:{:.2f}%".format(acc*100))

    """SVM"""
    sgd_clf = SGDClassifier()
    sgd_clf.fit(train_sents_tfidf,train_labels)
    predicted = sgd_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted==np.array(test_labels))
    print("ACC SVM:{:.2f}%".format(acc*100))

    """k近邻"""
    kn_clf = KNeighborsClassifier()
    kn_clf.fit(train_sents_tfidf,train_labels)
    predicted = kn_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted == np.array(test_labels))
    print("ACC KNN:{:.2f}%".format(acc * 100))

    """随机森林"""
    rf_clf = RandomForestClassifier()
    rf_clf.fit(train_sents_tfidf,train_labels)
    predicted = rf_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted == np.array(test_labels))
    print("ACC RF:{:.2f}%".format(acc * 100))

    """kmeans"""
    km_clf = KMeans(n_clusters=2).fit(train_sents_tfidf)
    predicted = km_clf.predict(test_sents_tfidf)
    acc = np.mean(predicted == np.array(test_labels))
    print("ACC Kmeans:{:.2f}%".format(acc * 100))
if __name__ == "__main__":
    main()