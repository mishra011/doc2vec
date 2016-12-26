import gensim
import numpy as np

d2v_model = gensim.models.doc2vec.Doc2Vec.load('science_doc2vec.model')



#docvec = d2v_model.docvecs[1]
#print docvec


docvec = d2v_model.docvecs['1.txt']  # if string tag used in training
vector = np.array(docvec)
print vector.size
"""
sims = d2v_model.docvecs.most_similar(14) # doc-id in bracket
#print sims

sims = d2v_model.docvecs.most_similar('1.txt')
#print sims

sims = d2v_model.docvecs.most_similar(docvec)
print sims


"""

docvec = d2v_model.docvecs['war.txt']
vector = np.array(docvec)
print vector
