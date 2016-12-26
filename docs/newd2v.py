import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities
import gensim
from sklearn import svm, metrics
import numpy

#Read in texts into div_texts(for LDA and Doc2Vec)
div_texts = []
f = open("clean_ad_nonad.txt")
lines = f.readlines()
f.close()
for line in lines:
    div_texts.append(line.strip().split(" "))

#Set up dictionary and MMcorpus
dictionary = corpora.Dictionary(div_texts)
dictionary.save("ad_nonad_lda_deeplearning.dict")
#dictionary = corpora.Dictionary.load("ad_nonad_lda_deeplearning.dict")
print dictionary.token2id["junk"]
corpus = [dictionary.doc2bow(text) for text in div_texts]
corpora.MmCorpus.serialize("ad_nonad_lda_deeplearning.mm", corpus)

#LDA training
id2token = {}
token2id = dictionary.token2id
for onemap in dictionary.token2id:
    id2token[token2id[onemap]] = onemap
#ldamodel = models.LdaModel(corpus, num_topics = 100, passes = 1000, id2word = id2token)
#ldamodel.save("ldamodel1000pass.lda")
#ldamodel = models.LdaModel(corpus, num_topics = 100, id2word = id2token)
ldamodel = models.LdaModel.load("ldamodel1000pass.lda")
ldatopics = ldamodel.show_topics(num_topics = 100, num_words = len(dictionary), formatted = False)
print ldatopics[10][1]
print ldatopics[10][1][1]
ldawordindex = {}
for i in range(len(dictionary)):
    ldawordindex[ldatopics[0][i][1]] = i

#Doc2Vec initialize
sentences = []
for i in range(len(div_texts)):
    string = "SENT_" + str(i)
    sentence = models.doc2vec.LabeledSentence(div_texts[i], labels = [string])
    sentences.append(sentence)
doc2vecmodel = models.Doc2Vec(sentences, size = 100, window = 5, min_count = 0, dm = 1)
print "Initial word vector for word junk:"
print doc2vecmodel["junk"]

#Replace the word vector with word vectors from LDA
print len(doc2vecmodel.syn0)
index2wordcollection = doc2vecmodel.index2word
print index2wordcollection
for i in range(len(doc2vecmodel.syn0)):
    if index2wordcollection[i].startswith("SENT_"):
        continue
    wordindex = ldawordindex[index2wordcollection[i]]
    wordvectorfromlda = [ldatopics[j][wordindex][0] for j in range(100)]
    doc2vecmodel.syn0[i] = wordvectorfromlda
#print doc2vecmodel.index2word[26841]
#doc2vecmodel.syn0[0] = [0 for i in range(100)]
print "Changed word vector for word junk:"
print doc2vecmodel["junk"]

#Train Doc2Vec
doc2vecmodel.train_words = False 
print "Initial doc vector for 1st document"
print doc2vecmodel["SENT_0"]
for i in range(50):
    print "Round: " + str(i)
    doc2vecmodel.train(sentences)
print "Trained doc vector for 1st document"
print doc2vecmodel["SENT_0"]

#Using SVM to do classification
resultlist = []
for i in range(4143):
    string = "SENT_" + str(i)
    resultlist.append(doc2vecmodel[string])
svm_x_train = []
for i in range(1000):
    svm_x_train.append(resultlist[i])
for i in range(2210,3210):
    svm_x_train.append(resultlist[i])
print len(svm_x_train)

svm_x_test = []
for i in range(1000,2210):
    svm_x_test.append(resultlist[i])
for i in range(3210,4143):
    svm_x_test.append(resultlist[i])
print len(svm_x_test)

svm_y_train = numpy.array([0 for i in range(2000)])
for i in range(1000,2000):
    svm_y_train[i] = 1
print svm_y_train

svm_y_test = numpy.array([0 for i in range(2143)])
for i in range(1210,2143):
    svm_y_test[i] = 1
print svm_y_test


svc = svm.SVC(kernel='linear')
svc.fit(svm_x_train, svm_y_train)

expected = svm_y_test
predicted = svc.predict(svm_x_test)

print("Classification report for classifier %s:\n%s\n"
      % (svc, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print doc2vecmodel["junk"]
