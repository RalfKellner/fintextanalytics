import os
import pickle
import nltk
import umap
import hdbscan
import warnings
import pandas as pd
import numpy as np
from sklearn.cluster import dbscan
from sklearn.preprocessing import normalize
from .utils import read_word_list, read_pickle
from gensim.models.doc2vec import Doc2Vec

class EmbeddingAnalyzer:

    def __init__(self):
        #self.doc_model = self._load_doc_model(doc_model_name)
        try: 
            self.doc_model = self._load_doc_model()
        except:
            self.doc_model = Doc2Vec.load('https://windat.uni-passau.de/filr/public-link/file-download/ff808082886b7c6001886c8bc5c4010c/11239/-6674408658168774027/d2v_10K_long.pkl')
            this_dir, _ = os.path.split(__file__)
            data_path = os.path.join(this_dir, "data", "d2v_10K_long.pkl")
            self.doc_model.save(data_path)

        self.word_vectors = self.doc_model.wv.get_normed_vectors()
        # uncomment if you want to train a 2D embedder for visualization
        #self.twod_embedder = umap.UMAP(n_neighbors =  15, n_components = 2, metric = 'cosine').fit(self.word_vectors)
        self.manual_topic_vectors = {}
        self.vocab = list(self.doc_model.wv.key_to_index.keys())
        self.word_list = self._load_word_list()


    def fit_topic_model(self, document_vectors, ntopic_words = 25, umap_args = None, hdbscan_args = None, rm_docs = False, cosine_threshold = 0.50, n_reduced_topics = None):
        
        '''
            This method uses a set of l2-normalized embedded document vectors, reduces their dimension by the UMAP dimensionality reduction model, 
            then the HDBSCAN cluster model to generate clusters which are considered to represent the topics, however, topic vectors are calculated using
            the centroid of embedded document vectors per cluster in the original embedding space. In addition, the number of topics is reduced
            by a cosine similarity threshold. After the process is finished, two sets of topic vectors and topic words are available. The one with the original
            topic number from the HDBSCAN algorithm and the one with a reduced number of topics.

            Parameters:
            ------------
            document_vectors : np.array
                a numpy array with l2-normalized document vector embeddings which match to the class doc2vec model
            ntopic_words: int
                the number of words which are most similar to the topic
            umap_args: dict
                a dictionary with parameters for the UMAP model, default=None using the internal default arguments
            hdbscan_args: dict
                a dictionary with parameters for the HDBSCAN model, default=None using the internal default arguments
            rm_docs: boolean
                save the document vectors which have been used to train the topics if True, default = False which means vectors are deleted 
                after topics have been trained
            cosine_threshold: fload
                a value between 0 and 1 which defines the threshold above which original topic vectors are grouped together when creating reduced
                 topic vectors
            n_reduced_topics: int
                can be used as an alternative to the cosine similarity logic for topic reduction, if a fixed number of topics is desired; 
                this should be done with caution, because topics with low similarity might be merged
        '''
        
        assert (cosine_threshold == None) + (n_reduced_topics == None) == 1, 'cosine_threshold or n_reduced_topics should be speficied, not both at the same time. The second argument must be None.' 
        if cosine_threshold:
            assert 0.10 <= cosine_threshold <= 0.90, 'cosine_threshold should lie in the range [0.10, 0.90] to get reasonable reduced topics.'
        else:
            assert isinstance(n_reduced_topics, int) and n_reduced_topics > 0, 'the number of n_reduced_topics must be a positive integer value'

        
        self.document_vectors = self._validate_normed_vectors(document_vectors)        
        self.ntopic_words = ntopic_words

        if umap_args == None:
            umap_args = {'n_neighbors': 15,
                'n_components': 5,
                'metric': 'cosine'}

        if hdbscan_args == None:
            hdbscan_args = {'min_cluster_size': 50,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom'}
            
        print('Starting to train the dimensionality reduction model...')
        self.umap_model = umap.UMAP(**umap_args).fit(self.document_vectors)
        print('Umap model for dimensionality reduction has been trained!\n')

        print('Starting to train the clustering model...')
        self.cluster_model = hdbscan.HDBSCAN(**hdbscan_args).fit(self.umap_model.embedding_)
        print('HDBSCAN model for clustering has been trained!\n')

        print('Creating topic vectors...')
        self._create_topic_vectors(self.cluster_model.labels_)
        self._deduplicate_topics()
        print(f'Creating topic vectors finished! {self.raw_topic_vectors.shape[0]} topics have been found.')
        print('Creating a reduced topic model...')
        self._reduce_topic_vectors(cosine_threshold = cosine_threshold, n_reduced_topics = n_reduced_topics)
        print(f'Topic reduction is finished, the number of reduced topics is {self.topic_vectors.shape[0]}')

        self.raw_topic_words, self.raw_topic_word_scores = self._find_topic_words_and_scores(topic_vectors=self.raw_topic_vectors, nwords = self.ntopic_words)
        self.topic_words, self.topic_word_scores = self._find_topic_words_and_scores(topic_vectors=self.topic_vectors, nwords = self.ntopic_words)

        if rm_docs:
            self.document_vectors = None


    def find_close_docs_to_words(self, document_vectors, wordlist = 'esgwords', topn_documents = 5):
        '''
        This function looks for documents whose meaning is close to the words provided.

        Parameters:
        ------------
        document_vectors: list or array of embedded document vectors
        wordlist: str or list
            If you want to use internal word lists the string must be either: ewords, swords, gwords or esgwords. Or, you simply provide a list
            of strings, e.g., ['expect', 'gdp', 'revenue']
        topn_documents: int
            Specficy the outputted number of close document ids to words

        Returns: 
        ---------
        tuple
            a tuple of two dataframes, the first with close document ids, the second with corresponding cosine similarities
        '''

        document_vectors_valid = self._validate_normed_vectors(document_vectors)
        self._validate_words(wordlist)
        not_in_vocab, in_vocab = self._wordlist_prepared(wordlist)

        normed_word_vectors = np.array([self._l2_normalize(vec) for vec in [self._generate_avg_word_vector(word) if isinstance(word, list) else self.doc_model.wv[word] for word in in_vocab]])
        #normed_word_vectors = np.array([self._l2_normalize(self.doc_model.wv[word]) for word in in_vocab])
        
        res = np.inner(normed_word_vectors, document_vectors_valid)
        top_docs = np.flip(np.argsort(res, axis = 1), axis = 1)[:, :topn_documents]
        top_doc_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, :topn_documents]

        in_vocab_index = [' '.join(word) if isinstance(word, list) else word for word in in_vocab]
        top_docs_df = pd.DataFrame(top_docs, index = in_vocab_index)
        top_doc_scores_df = pd.DataFrame(top_doc_scores, index = in_vocab_index)

        return top_docs_df, top_doc_scores_df
    

    def find_close_words_to_docs(self, document_vectors, wordlist = 'esgwords', topn_words = 5):
        '''
        This function looks for words whose meaning is close to the meaning of the documents.

        Parameters:
        ------------
        document_vectors: list or array of embedded document vectors
        wordlist: str or list
            If you want to use internal word lists the string must be either: ewords, swords, gwords or esgwords. Or, you simply provide a list
            of strings, e.g., ['expect', 'gdp', 'revenue']
        topn_words: int
            Specficy the outputted number of close words to document ids

        Returns:
        ---------
        tuple
            a tuple of two dataframes, the first with close words to document ids, the second with corresponding cosine similarities            
        '''

        document_vectors_valid = self._validate_normed_vectors(document_vectors)
        self._validate_words(wordlist)
        not_in_vocab, in_vocab = self._wordlist_prepared(wordlist)
    
        normed_word_vectors = np.array([self._l2_normalize(vec) for vec in [self._generate_avg_word_vector(word) if isinstance(word, list) else self.doc_model.wv[word] for word in in_vocab]])
        res = np.inner(document_vectors_valid, normed_word_vectors)
        top_words = np.flip(np.argsort(res, axis = 1), axis = 1)[:, :topn_words]
        top_word_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, :topn_words]

        top_words_df = pd.DataFrame(top_words)
        for i in range(top_words_df.shape[0]):
            top_words_df.iloc[i, :] = [in_vocab[idx] for idx in list(top_words_df.iloc[i].values)]
        top_word_scores_df = pd.DataFrame(top_word_scores)

        return top_words_df, top_word_scores_df
    

    def find_close_topics_to_words(self, topic_vectors, wordlist = 'esgwords', topn_topics = 3):
        '''
        This function looks for topics which are close to the meaning of certain words.

        Parameters:
        ------------
        topic_vectors: list or array of topic vectors; Note: topic vectors must be learned by the fit_topic_model method
        wordlist: str or list
            If you want to use internal word lists the string must be either: ewords, swords, gwords or esgwords. Or, you simply provide a list
            of strings, e.g., ['expect', 'gdp', 'revenue']
        topn_topics: int
            Specficy the outputted number of close topics to words

        Returns:
        ---------
        tuple
            a tuple of two dataframes, the first with close topics to words, the second with corresponding cosine similarities
        '''
        topic_vectors_valid = self._validate_normed_vectors(topic_vectors)
        self._validate_words(wordlist)
        not_in_vocab, in_vocab = self._wordlist_prepared(wordlist)
        
        normed_word_vectors = np.array([self._l2_normalize(vec) for vec in [self._generate_avg_word_vector(word) if isinstance(word, list) else self.doc_model.wv[word] for word in in_vocab]])
        res = np.inner(normed_word_vectors, topic_vectors_valid)
        top_topics = np.flip(np.argsort(res, axis = 1), axis = 1)[:, :topn_topics]
        top_topic_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, :topn_topics]

        in_vocab_index = [' '.join(word) if isinstance(word, list) else word for word in in_vocab]
        top_topics_df = pd.DataFrame(top_topics, index = in_vocab_index)
        top_topic_scores_df = pd.DataFrame(top_topic_scores, index = in_vocab_index)

        return top_topics_df, top_topic_scores_df


    def find_close_topics_to_docs(self, document_vectors, topic_vectors, topn_topics = 3):
        '''
        This function looks for topics which are close to documents.

        Parameters:
        ------------
        document_vectors: list or array of embedded document vectors
        topic_vectors: list or array of topic vectors; Note: topic vectors must be learned by the fit_topic_model method
        topn_topics: int
            Specficy the outputted number of close topics to documents

        Returns:
        ---------
        tuple
            a tuple of two dataframes, the first with close topics to documents, the second with corresponding cosine similarities
        '''
        document_vectors_valid = self._validate_normed_vectors(document_vectors)
        topic_vectors_valid = self._validate_normed_vectors(topic_vectors)
        
        res = np.inner(document_vectors_valid, topic_vectors_valid)
        top_topics = np.flip(np.argsort(res, axis = 1), axis = 1)[:, :topn_topics]
        top_topic_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, :topn_topics]

        top_topics_df = pd.DataFrame(top_topics)
        top_topic_scores_df = pd.DataFrame(top_topic_scores)

        return top_topics_df, top_topic_scores_df


    def find_close_docs_to_topics(self, topic_vectors, document_vectors, topn_docs = 3):
        '''
        This function looks for documents which are close to topics.

        Parameters:
        ------------
        document_vectors: list or array of embedded document vectors
        topic_vectors: list or array of topic vectors; Note: topic vectors must be learned by the fit_topic_model method
        topn_topics: int
            Specficy the outputted number of close topics to documents

        Returns:
        ---------
        tuple
            a tuple of two dataframes, the first with close topics to documents, the second with corresponding cosine similarities
        '''
        document_vectors_valid = self._validate_normed_vectors(document_vectors)
        topic_vectors_valid = self._validate_normed_vectors(topic_vectors)
        
        res = np.inner(topic_vectors_valid, document_vectors_valid)
        top_docs = np.flip(np.argsort(res, axis = 1), axis = 1)[:, :topn_docs]
        top_doc_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, :topn_docs]

        top_docs_df = pd.DataFrame(top_docs)
        top_doc_scores_df = pd.DataFrame(top_doc_scores)

        return top_docs_df, top_doc_scores_df
    

    def most_similar_words(self, wordlist = 'esgwords', n_words = 5):
        '''
        This function returns words which have similar meaning to words in the wordlist.

        Parameters:
        ------------
        wordlist: str or list
            If you want to use internal word lists the string must be either: ewords, swords, gwords or esgwords. 
            Or, you simply provide a list of strings, e.g., ['expect', 'gdp', 'revenue']

        n_words: int
            the number of words with similar context

        Returns:
        ---------
        tuple
            a tuple of two dataframes, the first with words close to words, the second with corresponding cosine similarities

        '''
        
        self._validate_words(wordlist)
        not_in_vocab, in_vocab = self._wordlist_prepared(wordlist)

        normed_word_vectors = np.array([self._l2_normalize(vec) for vec in [self._generate_avg_word_vector(word) if isinstance(word, list) else self.doc_model.wv[word] for word in in_vocab]])
        res = np.inner(normed_word_vectors, self.word_vectors)

        most_sim_idx = np.flip(np.argsort(res, axis = 1), axis = 1)[:, 1:(n_words+1)]
        most_sim_words = [[self.doc_model.wv.index_to_key[idx] for idx in list(idx_array)] for idx_array in list(most_sim_idx)]
        most_sim_scores = np.flip(np.sort(res, axis = 1), axis = 1)[:, 1:(n_words+1)]
        
        most_sim_words_df = pd.DataFrame(most_sim_words, index = in_vocab)
        most_sim_scores_df = pd.DataFrame(most_sim_scores, index = in_vocab)

        return most_sim_words_df, most_sim_scores_df
    

    def identify_outlier_word(self, wordlist):
        '''
        This function can be used to identify if certain words of a wordlist exhibit low similarity on average to the remaining words.


        Parameters:
        -------------
        wordlist: a list of str

        Returns:
        ---------
        exclude_words: array with all words sorted increasing by lowest average similarity
        exclude_word_scores: array with average cosine similarities of this word to the remaining words


        '''

        not_in_vocab, in_vocab = self._wordlist_prepared(wordlist)

        word_index = [self.doc_model.wv.key_to_index[word] for word in in_vocab]
        word_embeddings = self.word_vectors[word_index]
        res = np.inner(word_embeddings, word_embeddings)
        exclude_words = [in_vocab[i] for i in np.argsort(res.mean(axis = 1))]
        exclude_word_scores = np.sort(res.mean(axis = 1))

        return exclude_words, exclude_word_scores


    def create_wordbased_topic(self, wordlist, name):
        '''
        This function can be used to add a self-defined topic vector which is the average of word vectors in the word_list
        
        Parameters:
        -------------
        wordlist: a list of str
        name: str which serves as an identifier such that the build topic vector can be found in the self.manual_topic_vectors dictionary

        Returns:
        ---------

        '''

        not_in_vocab, in_vocab = self._wordlist_prepared(wordlist)

        word_index = [self.doc_model.wv.key_to_index[word] for word in in_vocab]
        word_embeddings = self.word_vectors[word_index]
        self.manual_topic_vectors[name] = self._l2_normalize(word_embeddings).mean(axis = 0).reshape(1, -1)


    def save(self, file):
        '''
        This function can be used to store a trained EmbeddingAnalyzer instance as a pickle file.
        '''
        #create a pickle file
        with open(file, 'wb') as handle:
            #pickle the dictionary and write it to file
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load(self, file):
        '''
        This function can be used to load a previously trained EmbeddingAnalyzer instance
        '''
        with open(file, 'rb') as handle:
            out = pickle.load(handle)
        return out


    def _find_topic_words_and_scores(self, topic_vectors, nwords):
        '''
        This function is made for internal usage.
        '''

        topic_words = []
        topic_word_scores = []

        res = np.inner(topic_vectors, self.word_vectors)

        top_words = np.flip(np.argsort(res, axis = 1), axis = 1)
        top_scores = np.flip(np.sort(res, axis = 1), axis = 1)

        for words, scores in zip(top_words, top_scores):
            topic_words.append([self.vocab[i] for i in words[0:nwords]])
            topic_word_scores.append(scores[0:nwords])

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)

        return topic_words, topic_word_scores
    

    def _create_topic_vectors(self, cluster_labels):
        '''
        This function is made for internal usage.
        '''

        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        self.raw_topic_vectors = self._l2_normalize(
            np.vstack([self.document_vectors[np.where(cluster_labels == label)[0]]
                      .mean(axis=0) for label in unique_labels]))
        
    
    def _deduplicate_topics(self):
        '''
        This function is made for internal usage.
        '''

        core_samples, labels = dbscan(X=self.raw_topic_vectors,
                                      eps=0.1,
                                      min_samples=2,
                                      metric="cosine")

        duplicate_clusters = set(labels)

        if len(duplicate_clusters) > 1 or -1 not in duplicate_clusters:

            unique_topics = self.raw_topic_vectors[np.where(labels == -1)[0]]

            if -1 in duplicate_clusters:
                duplicate_clusters.remove(-1)

            for unique_label in duplicate_clusters:
                unique_topics = np.vstack(
                    [unique_topics, self._l2_normalize(self.raw_topic_vectors[np.where(labels == unique_label)[0]]
                                                       .mean(axis=0))])

            self.raw_topic_vectors = unique_topics
        

    def _reduce_topic_vectors(self, cosine_threshold, n_reduced_topics):
        '''
        This function is made for internal usage.
        '''

        topic_vectors = self.raw_topic_vectors
        n_topics = topic_vectors.shape[0]
        inner = np.inner(topic_vectors, topic_vectors)
        np.fill_diagonal(inner, 0.0)
        topsim = np.max(inner)

        if cosine_threshold:
            while topsim > cosine_threshold:
                inner = np.inner(topic_vectors, topic_vectors)
                np.fill_diagonal(inner, 0.0)
                topsim = np.max(inner)
                topic_pair, _ = np.where(inner == topsim)
                new_topic = topic_vectors[topic_pair].mean(axis = 0)
                topic_vectors = np.delete(topic_vectors, topic_pair, axis = 0)
                topic_vectors = np.append(topic_vectors, new_topic.reshape(1, -1), axis = 0)
        elif n_reduced_topics:
            while n_topics > n_reduced_topics:
                inner = np.inner(topic_vectors, topic_vectors)
                np.fill_diagonal(inner, 0.0)
                topsim = np.max(inner)
                topic_pair, _ = np.where(inner == topsim)
                new_topic = topic_vectors[topic_pair].mean(axis = 0)
                topic_vectors = np.delete(topic_vectors, topic_pair, axis = 0)
                topic_vectors = np.append(topic_vectors, new_topic.reshape(1, -1), axis = 0)
                n_topics = topic_vectors.shape[0]

        self.topic_vectors = self._l2_normalize(topic_vectors)

        print(f'The number of topics has been reduced from {len(self.raw_topic_vectors)} to {len(self.topic_vectors)}.')


    def _wordlist_prepared(self, words, print_info = True):
        '''
        This function is made for internal usage.
        '''

        if isinstance(words, str):
            word_list = self.word_list[words]
        elif isinstance(words, list):
            word_list = words

        in_vocab = []
        not_in_vocab = []
        for word in word_list:
            if isinstance(word, list):
                if all([w in self.vocab for w in word]):
                    in_vocab.append(word)
                else:
                    not_in_vocab.append(word)
            else:
                if word in self.vocab:
                    in_vocab.append(word)
                else:
                    not_in_vocab.append(word)

        if (len(not_in_vocab) > 0) and print_info:
            print('The following words are not in the vocabulary:\n')
            print(not_in_vocab)

        if len(in_vocab) == 0:
            raise ValueError('No word of the word list exists in the Word2Vec model.')
        
        return not_in_vocab, in_vocab
    

    def _load_word_list(self):
        '''
        This function is made for internal usage.
        '''

        word_list = dict()
        for filename in ["ewords.txt", "swords.txt", "gwords.txt"]:
            this_dir, _ = os.path.split(__file__)
            data_path = os.path.join(this_dir, "data", filename)
            word_list[filename.split(".")[0]] = read_word_list(data_path)
        all_words = []
        for key in word_list.keys():
            all_words += word_list[key]
        word_list['esgwords'] = all_words

        lost_words = []
        for key in word_list.keys():
            not_in_vocab, in_vocab = self._wordlist_prepared(word_list[key], print_info=False)
            word_list[key] = in_vocab
            lost_words.extend(not_in_vocab)
        print(f'In sum {len(set(lost_words))} words from the default word lists could not be found in the Word2Vec vocabulary. These words are deleted from the word list:')
        print(set(lost_words))
        return word_list
    
    
    def _validate_normed_vectors(self, vectors):
        '''
        This function is made for internal usage.
        '''

        vectors_out = vectors
        if isinstance(vectors_out, list):
            vectors_out = np.array(vectors_out)
        norm_check = any([not(0.999 <= nbr <= 1.001) for nbr in np.linalg.norm(vectors_out, axis = 1)])
        if norm_check:
            vectors_out = self._l2_normalize(vectors_out)
            print(f'Vectors seem not have unit length...the vectors have been l2-normalized.')
        
        return vectors_out
    

    def _generate_avg_word_vector(self, words):
        '''
        This function is made for internal usage. It calculates the average vector for a list of given word vectors
        '''
    
        avg_vector = np.zeros(self.doc_model.wv[0].shape[0])
        for word in words:
            avg_vector += self.doc_model.wv[word]
        avg_vector = avg_vector / len(words)
        return avg_vector


    @staticmethod
    def _l2_normalize(vectors):
        '''
        This function is made for internal usage.
        '''

        if vectors.ndim == 2:
            return normalize(vectors)
        else:
            return normalize(vectors.reshape(1, -1))[0]


    @staticmethod
    def _load_doc_model():
        '''
        This function is made for internal usage.
        '''

        this_dir, _ = os.path.split(__file__)
        data_path = os.path.join(this_dir, "data", "d2v_10K_long.pkl")
        
        model = Doc2Vec.load(data_path)
    
        return model


    @staticmethod
    def _validate_words(words):
        '''
        This function is made for internal usage.
        '''

        if isinstance(words, str):
            assert words in ['ewords', 'swords', 'gwords', 'esgwords'], 'When using internal word list, words must be "ewords", "swords", "gwords" or "esgwords"'
        elif isinstance(words, list):
            print('Identified user defined word list. Note this list should contain of strings or list of strings. The latter will be merged to one embedding vector by averaging the words.')
        else:
            raise TypeError('Either use a string for build-in word lists or provide a list of strings.')
