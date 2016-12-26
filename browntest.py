import gensim


model = gensim.models.Doc2Vec.load('brown_model')
#model = gensim.models.Word2Vec.load('brown_model')
print model.most_similar(positive=['women', 'adult'], negative=['man'])

