from gensim.models import CoherenceModel
# topic diversity metrics 
def compute_silhouette_score(X, labels):
  return silhouette_score(X, labels)

def calc_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words,[]))
    n_total = len(topic_words) * len(topic_words[0])
    topic_div = len(vocab) / n_total
    return topic_div

def calc_topic_coherence(topic_words,docs,dictionary,taskname=None,sents4emb=None,calc4each=False):
    # emb_path: path of the pretrained word2vec weights, in text format.
    # sents4emb: list/generator of tokenized sentences.
    # Computing the C_V score
    cv_coherence_model = CoherenceModel(topics=topic_words, texts=docs, dictionary=dictionary, coherence='c_v')
    cv_per_topic = cv_coherence_model.get_coherence_per_topic() if calc4each else None
    cv_score = cv_coherence_model.get_coherence()
    
    # Computing the C_W2V score
    try:
        w2v_model_path = os.path.join(os.getcwd(),'data',f'{taskname}','w2v_weight_kv.txt')
        # Priority order: 1) user's embed file; 2) standard path embed file; 3) train from scratch then store.
        if emb_path!=None and os.path.exists(emb_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(emb_path,binary=True)
        elif os.path.exists(w2v_model_path):
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path,binary=True)
        w2v_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_w2v',keyed_vectors=keyed_vectors)
        w2v_per_topic = w2v_coherence_model.get_coherence_per_topic() if calc4each else None
        w2v_score = w2v_coherence_model.get_coherence()
    except Exception as e:
        print(e)
        #In case of OOV Error
        w2v_per_topic = [None for _ in range(len(topic_words))]
        w2v_score = None
    
    # Computing the C_UCI score
    c_uci_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_uci')
    c_uci_per_topic = c_uci_coherence_model.get_coherence_per_topic() if calc4each else None
    c_uci_score = c_uci_coherence_model.get_coherence()
    
    
    # Computing the C_NPMI score
    c_npmi_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_npmi')
    c_npmi_per_topic = c_npmi_coherence_model.get_coherence_per_topic() if calc4each else None
    c_npmi_score = c_npmi_coherence_model.get_coherence()
    return (cv_score,w2v_score,c_uci_score, c_npmi_score),(cv_per_topic,w2v_per_topic,c_uci_per_topic,c_npmi_per_topic)



#same as in calc_topic_coherence
def calc_cw2v(topic_words,docs,dictionary,emb_path=None):
  keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(emb_path,binary=True)
  w2v_coherence_model = CoherenceModel(topics=topic_words,texts=docs,dictionary=dictionary,coherence='c_w2v',keyed_vectors=keyed_vectors)
  w2v_score = w2v_coherence_model.get_coherence()
  return w2v_score




#I didn't use this function
def calc_similarity_sentroid(topic_list, word2vec_path, binary = True, topk = 10):
  wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)
  sim = 0
  count = 0
  for list1, list2 in combinations(topic_list, 2):
    centroid1 = np.zeros(wv.vector_size)
    centroid2 = np.zeros(wv.vector_size)
    count1, count2 = 0, 0
    for word1 in list1[:topk]:
      if word1 in wv.key_to_index.keys():
        centroid1 = centroid1 + wv[word1]
        count1 += 1
        for word2 in list2[:topk]:
          if word2 in wv.key_to_index.keys():
            centroid2 = centroid2 + wv[word2]
            count2 += 1
            centroid1 = centroid1 / count1
            centroid2 = centroid2 / count2
            sim = sim + (1 - cosine(centroid1, centroid2))
            count += 1
  return sim / count
