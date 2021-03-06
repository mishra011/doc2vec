#python example to infer document vectors from trained doc2vec model
import gensim.models as g
#import codecs
import numpy as np
#parameters
model="science_doc2vec.model"
test_docs="docs/war.txt"
#output_file="toy_data/test_vectors.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
m = g.Doc2Vec.load(model)
test = []
test.append(open(test_docs).readline())
#print test

#infer test vectors
#output = open(output_file, "w")
vec = []
vec.append([str(x) for x in m.infer_vector(test, alpha=start_alpha, steps=infer_epoch)])
a = np.array(vec)
print a.shape
print test
