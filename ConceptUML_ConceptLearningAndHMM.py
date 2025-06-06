'''May 2025
Add HMM_Batch_RecordTimeandMem to record time and mem.

@author: Khanh Luong
20 Dec 23:
    Re-tokenizing mitre and capec to ensure techniques are kept, .exe are kept.
    Using new version of Mitre_tokens file and capec token file as well as new data file.

'''
import os
import time
import re
import pickle
import psutil
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from collections import Counter
from transformers import BertTokenizer
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import dask.array as da
from scipy import stats
import pickle
from Libraries_and_Funcs.Evaluation import calc_topic_diversity, calc_topic_coherence

path = ''
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def apply_command_line_rules(command_line):
    rules = {
        r"[,]": r"",
        r"[ ]{2,}": r" ",
        r'"(.+?)"': r"\1",
        r"\'(.+?)\'": r"\1",
        r"^\\\?\?\\(c:.+?$)": r"\1",
        r"c:\\.+\\(.+?\.exe)": r"\1",
        r"(-[a-z]{1,}) \d{1,}": r"\1 num",
        r"(?:(?:http[s]?:\/\/)|www\.)(?:www\.)?[^\s\.]{1,}\.[^\s]{2,}": "url",
        r"\S+@\S+\.\S+": "email",
        r"(?:(?:25[0-5]|(?:2[0-4]|1\d|[1-9]|)\d)\.){3}(?:25[0-5]|(?:2[0-4]|1\d|[1-9]|)\d)": "ip",
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}": r"uuid",
    }

    for pattern, replacement in rules.items():
        command_line = re.sub(pattern, replacement, command_line)
    
    command_line = re.sub(r'\s+', ' ',command_line)
        
    return command_line

def prepare_datasets(path, num_samps, num_train_samps,training_ratio = 0.8, filename="parsed_data_large_annotated.csv"):
    data = pd.read_csv(path + 'Data/'+filename)[:num_samps]

    df = data[['UtcTime','RuleName','ProcessGuid', 'ProcessId', 'User','EventCode','EventType',
           'TaskCategory=Process Create (rule','FileVersion','Description','OriginalFileName', 
           'CommandLine','LogonGuid','LogonId','IntegrityLevel',
           'ParentProcessGuid','ParentProcessId','ParentCommandLine',
           'TaskCategory=Registry value set (rule','TargetObject','MD5','SHA256','IMPHASH','Label']].copy()

    #df['Event'] = df.drop(columns=['Label']).apply(lambda row: ' '.join(map(str, row)), axis=1)#KK can keep this for original context
    #'ParentImage' and Image are contained in CommandLine and ParentCommandLine so not need
    df['CommandLine'] = df['CommandLine'].str.lower()
    df['CommandLine'].fillna(' ', inplace=True)
    df['Processed_CommandLine'] = df['CommandLine'].apply(apply_command_line_rules)
    
    df['ParentCommandLine'].fillna(' ', inplace=True)
    df['ParentCommandLine'] = df['ParentCommandLine'].str.lower()
    df['Processed_ParentCommandLine'] = df['ParentCommandLine'].apply(apply_command_line_rules)
    df['Event'] = df.drop(columns=['Label','CommandLine','ParentCommandLine']).apply(lambda row: ' '.join(map(str, row)), axis=1)
    
    #df.info()
    #seq_len = 10
    df_seq = create_event_seq(df)
    df_seq.to_excel("df_seq.xlsx")
    main_text = df_seq['Event']
    
    lower_text = []
    for text in main_text:
        n_text = text.lower()
        lower_text.append(n_text)

    filtered_texts = []
    for text in lower_text:
        # Remove all the special characters
        text = re.sub(r'nan', '', text)
        filtered_texts.append(text)

    df_seq['processed_text'] = filtered_texts
    main_df = df_seq[['Event', 'processed_text', 'Label', 'Seq_len']].copy()
    num_samps = main_df.shape[0]
    if training_ratio !=0:
        num_train_samps = int(num_samps * training_ratio)
        print("num_train_samps = ",num_train_samps)
    train_df = main_df[:num_train_samps]     
    test_df = main_df[num_train_samps:]
    return main_df, train_df, test_df, data

def create_event_seq(df, seq_len):
    combined_events = []
    max_labels = []

    current_events = []
    current_max_label = None

    for index, row in df.iterrows():
        event = row['Event']
        label = row['Label']

        if len(current_events) < seq_len:
            current_events.append(event)

            if current_max_label is None or label > current_max_label:
                current_max_label = label
        else:
            combined_events.append(' '.join(current_events))
            max_labels.append(current_max_label)

            current_events = [event]
            current_max_label = label

    # Append the last group to the lists
    combined_events.append(' '.join(current_events))
    max_labels.append(current_max_label)

    df_seq = pd.DataFrame({'Event': combined_events, 'Label': max_labels})

    return df_seq
def prepare_model_input(train_df):
    events = train_df['Event'].str.strip().tolist()
    sp = WhiteSpacePreprocessing(events, "english")
    processed_events, unprocessed_corpus, vocab = sp.preprocess()
    
    filtered_texts = []
    for text in processed_events:
        # Remove all the special characters
        text = re.sub(r'nan', '', text)
        filtered_texts.append(text)
    processed_events = filtered_texts
    
    text_tokens = [doc.split() for doc in processed_events]
    
    # Create gensim dictionary and corpus
    df_dic = Dictionary(text_tokens)
    tfidf = TfidfModel(dictionary=df_dic)
    train_corpus = [df_dic.doc2bow(document) for document in text_tokens]
    train_corpus_tfidf = list(tfidf[train_corpus])
    return processed_events, unprocessed_corpus, text_tokens, train_corpus_tfidf, df_dic

# =============================================================================
def preprocess_data(current_df, column_name = "Event"):
    # Initialize CountVectorizer
    vectorizer = CountVectorizer()
    
    # Fit and transform the unprocessed_corpus
    bow_embeddings = vectorizer.fit_transform(current_df[column_name])
    bow_embeddings = bow_embeddings.toarray()
    
    # vocabulary = vectorizer.get_feature_names_out()
    all_tokens = [token for tokens_list in current_df['cleaned_tokens'] for token in tokens_list]
    vocabulary = set(all_tokens)
    

    # Initialize SentenceTransformer model (e.g., 'bert-base-nli-mean-tokens')
    contextualized_model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    if 'processed_text' not in current_df.columns:
        current_df['processed_text'] = current_df[column_name].str.lower() # add other preprocessing steps as needed
       
    # Encode the processed_events to get contextualized embeddings
    
    start_time = time.time()
    contextualized_embeddings = contextualized_model.encode(current_df['cleaned_tokens'])
    bert_time = time.time() - start_time
    print(f"BERT embedding time: {bert_time:.2f} seconds")
    
    bert_mem = get_memory_usage_mb()
    print(f"BERT Embedding Time: {bert_time:.2f}s | Memory: {bert_mem:.2f}MB")
    
    contextualized_embeddings = np.array(contextualized_embeddings)

    # Concatenate BoW and contextualized embeddings horizontally (axis=1)
    concatenated_embeddings = np.concatenate((bow_embeddings, contextualized_embeddings), axis=1)

    # Scale/normalize the concatenated embeddings
    concatenated_embeddings_pos = scale_or_normalize(concatenated_embeddings, method="min-max")
    
    
    return concatenated_embeddings_pos, vocabulary, bow_embeddings, contextualized_embeddings,bert_time,bert_mem
def save_hmm_kmeans_results_to_excel_1(batch_df, method, prediction_col_name, dataset_name, num_clusters, 
                                       max_mean_cosine_similarity, max_mean_cosine_index, 
                                       mean_cosine_similarity, max_capec_mean_cosine_similarity, 
                                       max_capec_mean_cosine_index, max_combined_score, 
                                       capec_mean_cosine_similarity, max_combined_index, 
                                       combined_score, score_likehoodOrsilhou, 
                                       nmf_topic_matrix, mitre_nmf_topic_matrix, capec_nmf_topic_matrix):
    """
    Saves HMM batch results to an Excel file.

    Args:
        batch_df (DataFrame): The batch DataFrame.
        method (str): 'HMM' or 'KMeans' to label the file.
        prediction_col_name (str): Column name with cluster labels.
        dataset_name (str): Name of the dataset.
        num_clusters (int): Number of clusters.
        Other metrics: similarity scores and evaluation results.
    """

    # Create the filename for the Excel file
    excel_file_name_hmm = f'{method}_{dataset_name}_{num_clusters}clusters.xlsx'

    # Create DataFrames for each set of results
    batch_df_df = pd.DataFrame(batch_df)
    mean_cosine_similarity_df = pd.DataFrame({'Mean_Cosine_Similarity': mean_cosine_similarity})
    capec_mean_cosine_similarity_df = pd.DataFrame({'CAPEC_Mean_Cosine_Similarity': capec_mean_cosine_similarity})
    combined_score_df = pd.DataFrame({'Combined_Score': combined_score})
    max_mean_cosine_similarity_df = pd.DataFrame({'Max_Mean_Cosine_Similarity': [max_mean_cosine_similarity]})
    max_mean_cosine_index_df = pd.DataFrame({'Max_Mean_Cosine_Index': [max_mean_cosine_index]})
    max_capec_mean_cosine_similarity_df = pd.DataFrame({'Max_CAPEC_Mean_Cosine_Similarity': [max_capec_mean_cosine_similarity]})
    max_capec_mean_cosine_index_df = pd.DataFrame({'Max_CAPEC_Mean_Cosine_Index': [max_capec_mean_cosine_index]})
    score_likehoodOrsilhou_df = pd.DataFrame({'Likehood_or_Silhou_Score': [score_likehoodOrsilhou]})

    hidden_states = batch_df_df[prediction_col_name]
    num_hid_states = len(np.unique(hidden_states))
    
    cluster_indices_df = pd.DataFrame({'Index': np.arange(num_hid_states)})

    dataframes = {
        'Index': cluster_indices_df,        
        'Mean_Cosine_Similarity': mean_cosine_similarity_df,
        'CAPEC_Mean_Cosine_Similarity': capec_mean_cosine_similarity_df,
        'Combined_Score': combined_score_df,
        'Max_Mean_Cosine_Similarity': max_mean_cosine_similarity_df,
        'Max_Mean_Cosine_Index': max_mean_cosine_index_df,
        'Max_CAPEC_Mean_Cosine_Similarity': max_capec_mean_cosine_similarity_df,
        'Max_CAPEC_Mean_Cosine_Index': max_capec_mean_cosine_index_df,
        'Score_likehoodOrsilhou': score_likehoodOrsilhou_df
    }
    combined_sheet_df = pd.concat(dataframes.values(), axis=1)

    # Compute cluster summary
    cluster_summary = batch_df_df.groupby(prediction_col_name).agg(
        total_samples=(prediction_col_name, 'size'),
        attack_event_count=('Label', lambda x: (x != 0).sum())
    ).reset_index()
    cluster_summary.columns = ['Cluster_ID', 'Cluster_Count', 'Attack_Event_Count']
    
    # Suspicious Cluster Ranking
    cluster_id = max_combined_index
    q = 500  # Top samples to identify
    best_num_clusters_df = batch_df_df.reset_index()
    best_num_clusters_df['mi_ca_label'] = 0

    cluster_indices = np.where(best_num_clusters_df[prediction_col_name] == cluster_id)[0]
    cluster_vectors = nmf_topic_matrix[cluster_indices]

    similarity_array_mitre = calculate_similarity(cluster_vectors, mitre_nmf_topic_matrix)
    similarity_array_capec = calculate_similarity(cluster_vectors, capec_nmf_topic_matrix)
    similarity_mitre_capec = similarity_array_mitre + similarity_array_capec

    ranks = np.argsort(np.argsort(-similarity_mitre_capec))

    # Update mitre_label for top samples in suspicious cluster
    for idx, original_index in enumerate(cluster_indices):
        rank = ranks[idx] + 1
        best_num_clusters_df.at[original_index, 'mi_ca_label'] = rank

    # Write results to Excel
    with pd.ExcelWriter(excel_file_name_hmm) as writer:
        batch_df_df.to_excel(writer, sheet_name='Batch_df', index=False)
        combined_sheet_df.to_excel(writer, sheet_name='All_Scores', index=False)
        cluster_summary.to_excel(writer, sheet_name='Statistics', index=False)
        best_num_clusters_df.to_excel(writer, sheet_name='SusClusterRanking', index=False)

    print("\n_________Done. Saving results to Excel file. ", excel_file_name_hmm)
    
def scale_or_normalize(current_data, method="min-max"):
    import dask.array as da
    # Convert to a Dask array with a smaller chunk size
    current_data = da.from_array(current_data, chunks=(500, 500))  # Adjust chunk sizes as appropriate

    if method == "min-max":
        # Using Dask's incremental computation for memory efficiency
        min_val = current_data.min().compute()
        max_val = current_data.max().compute()

        # Apply min-max scaling
        scaled_current_data = (current_data - min_val) / (max_val - min_val)
        scaled_current_data = scaled_current_data.compute()

        return scaled_current_data

    elif method == "z-score":
        # Using Dask's incremental computation for memory efficiency
        mean = current_data.mean().compute()
        std_dev = current_data.std().compute()

        # Apply z-score normalization
        z_score_normalized_current_data = (current_data - mean) / std_dev
        z_score_normalized_current_data = z_score_normalized_current_data.compute()

        return z_score_normalized_current_data

    else:
        raise ValueError("Invalid method. Use 'min-max' or 'z-score'.")

        
def count_samples_in_clusters(cluster_labels):
     cluster_count = Counter(cluster_labels)
     return cluster_count       
        

def custom_tokenize(text, tokenizer):
    vocabulary = set()
    pattern = r'\{[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\}'
    new_tokens = []
    split_text = re.split('('+pattern+')', text)
    for segment in split_text:
        if re.match(pattern, segment):
            new_tokens.append(segment)
            vocabulary.add(segment)
        else:
            words = segment.split()
            for word in words:
                if word in tokenizer.vocab:
                    tokens = tokenizer.tokenize(word)
                    new_tokens.extend(tokens)
                    vocabulary.update(tokens)
                else:
                    new_tokens.append(word)
                    vocabulary.add(word)

# =============================================================================
    return new_tokens
def prepare_data(batch_df, path, num_topics):        
    
    nmf_model = NMF(n_components=num_topics, random_state=42)
    print("\n_________Contextualized embeddings for batch_df")
   
    concatenated_embeddings_pos, vocabulary, bow_embeddings, contextualized_embeddings, bert_time, bert_mem = preprocess_data(batch_df)
# =============================================================================

    print("Normalizing the data....")
    contextualized_embeddings_pos = scale_or_normalize(contextualized_embeddings, method="min-max")
    
    start_time = time.time()
    nmf_topic_matrix = nmf_model.fit_transform(contextualized_embeddings_pos)
    nmf_time = time.time() - start_time
    print(f"NMF time: {nmf_time:.2f} seconds")
    nmf_mem = get_memory_usage_mb()
    print(f"NMF Time: {nmf_time:.2f}s | Memory: {nmf_mem:.2f}MB")

    print(contextualized_embeddings_pos[:5])
    print("\n_________Contextualized embeddings for mitre")
    mitre_df = pd.read_csv("Data/ExternalSources/MitreTechniquesTokens_V5.csv",encoding = "ISO-8859-1")
    
    # create tokens for mitre_df
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    
    mitre_concat_embeddings_pos, mitre_vocabulary, mitre_bow_embeddings, mitre_contextualized_embeddings, bert_time_mitre, bert_mem_mitre = preprocess_data(mitre_df, "description")
    mitre_contextualized_embeddings_pos = scale_or_normalize(mitre_contextualized_embeddings, method="min-max")

    #mitre_nmf_topic_matrix = nmf_model.fit_transform(mitre_bow_embeddings) 

    mitre_nmf_topic_matrix = nmf_model.fit_transform(mitre_contextualized_embeddings_pos) 
    
    print("\n_________Contextualized embeddings for capec")
    capec_df = pd.read_csv("Data/ExternalSources/CapecTokens_V5.csv",encoding = "ISO-8859-1")
    
    capec_concat_embeddings_pos, capec_vocabulary, capec_bow_embeddings, capec_contextualized_embeddings, bert_time_capec, bert_mem_capec = preprocess_data(capec_df, "Description")

    capec_contextualized_embeddings_pos = scale_or_normalize(capec_contextualized_embeddings, method="min-max")
    capec_nmf_topic_matrix = nmf_model.fit_transform(capec_contextualized_embeddings_pos) 
    return nmf_topic_matrix, batch_df, mitre_nmf_topic_matrix, capec_nmf_topic_matrix, capec_vocabulary, mitre_vocabulary, vocabulary, contextualized_embeddings_pos, mitre_contextualized_embeddings_pos, capec_concat_embeddings_pos, bert_time, bert_mem, nmf_time, nmf_mem
def kmeans_clustering(nmf_topic_matrix, num_clusters):
    from scipy.cluster.vq import kmeans, vq
    centroids, distortion = kmeans(nmf_topic_matrix, num_clusters)
    cluster_labels, _ = vq(nmf_topic_matrix, centroids)
    return cluster_labels, centroids

def calculate_and_save_similarity_scores(nmf_topic_matrix, mitre_nmf_topic_matrix, capec_nmf_topic_matrix, kmeans_labels, num_clusters, excel_file_path):
    print("\n_________Calculating the similarity score of each hidden state.")
    
    mean_cosine_similarity = []
    capec_mean_cosine_similarity = []
    combined_scores = []
    cluster_scores = []

    for cluster_id in range(num_clusters):
        cluster_indices = np.where(kmeans_labels == cluster_id)[0]
        cluster_vectors = nmf_topic_matrix[cluster_indices]

        # Calculate the cosine similarity for Mitre
        cosine_similarities = cosine_similarity(cluster_vectors, mitre_nmf_topic_matrix)
        mean_cosine_sim = np.mean(cosine_similarities, axis=1)
        mean_cosine_value = round(np.mean(mean_cosine_sim), 2)
        mean_cosine_similarity.append(mean_cosine_value)

        # Calculate the cosine similarity for CAPEC
        capec_cosine_similarities = cosine_similarity(cluster_vectors, capec_nmf_topic_matrix)
        capec_mean_cosine_sim = np.mean(capec_cosine_similarities, axis=1)
        capec_mean_cosine_value = round(np.mean(capec_mean_cosine_sim), 2)
        capec_mean_cosine_similarity.append(capec_mean_cosine_value)

        combined_score = mean_cosine_value + capec_mean_cosine_value
        combined_scores.append(combined_score)
        cluster_scores.append((f'Cluster {cluster_id}', mean_cosine_value, capec_mean_cosine_value, combined_score))

    # Find max values
    max_combined_score = max(combined_scores)
    max_combined_index = combined_scores.index(max_combined_score) + 1
    max_mean_cosine_similarity = max(mean_cosine_similarity)
    max_mean_cosine_index = mean_cosine_similarity.index(max_mean_cosine_similarity) + 1
    max_capec_mean_cosine_similarity = max(capec_mean_cosine_similarity)
    max_capec_mean_cosine_index = capec_mean_cosine_similarity.index(max_capec_mean_cosine_similarity) + 1

    # Save to Excel
    df_cluster_scores = pd.DataFrame(cluster_scores, columns=['Cluster', 'Mean Cosine Similarity', 'CAPEC Mean Cosine Similarity', 'Combined Score'])
    df_max_scores = pd.DataFrame({
        'Metric': ['Max Combined Score', 'Max Combined Index', 'Max Mean Cosine Similarity', 'Max Mean Cosine Index', 'Max CAPEC Mean Cosine Similarity', 'Max CAPEC Mean Cosine Index'],
        'Value': [max_combined_score, max_combined_index, max_mean_cosine_similarity, max_mean_cosine_index, max_capec_mean_cosine_similarity, max_capec_mean_cosine_index]
    })

    with pd.ExcelWriter(excel_file_path) as writer:
        df_cluster_scores.to_excel(writer, sheet_name='Cluster Scores', index=False)
        df_max_scores.to_excel(writer, sheet_name='Max Scores', index=False)

    return max_combined_score, max_combined_index, max_mean_cosine_similarity, max_mean_cosine_index, max_capec_mean_cosine_similarity, max_capec_mean_cosine_index

def HMM_Batch(batch_df, max_num_state, path,nmf_topic_matrix, mitre_nmf_topic_matrix, capec_nmf_topic_matrix, is_seq=1, num_topics=10):
    print("\n_________HMM learning")
    best_model = None
    best_log_likelihood = float('-inf')
    for n_states in range(4, max_num_state):
        for run in range(10):
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="tied")
            model.fit(nmf_topic_matrix)
            log_likelihood = model.score(nmf_topic_matrix)
            if log_likelihood > best_log_likelihood:
                best_model = model
                best_log_likelihood = log_likelihood
    
    hidden_states = best_model.predict(nmf_topic_matrix)
    batch_df['HMM_label'] = hidden_states
    cluster_count = count_samples_in_clusters(hidden_states)
    
    num_hid_states = len(np.unique(hidden_states))
    print("\n_________Done HMM. Number of hidden states: ",num_hid_states)
    for cluster, count in cluster_count.items():
        print(f"Cluster {cluster}: {count} samples")
    
    print("\n_________Calculating the similarity score of each hidden state.")    
    mean_cosine_similarity = []
    capec_mean_cosine_similarity = []
    for cluster_id in range(num_hid_states):
        cluster_indices = np.where(hidden_states == cluster_id)[0]
        cluster_vectors = nmf_topic_matrix[cluster_indices]        
        
        # Calculate the cosine similarity between Mitre and cluster_vectors
        cosine_similarities = cosine_similarity(mitre_nmf_topic_matrix, cluster_vectors)
        
        mean_cosine_sim = np.mean(cosine_similarities, axis=1)
        mean_cosine_value = round(np.mean(mean_cosine_sim),2)
        mean_cosine_similarity.append(mean_cosine_value)
        
        # Calculate the cosine similarity between CAPEC and cluster_vectors
        capec_cosine_similarities = cosine_similarity(capec_nmf_topic_matrix, cluster_vectors)
        
        capec_mean_cosine_sim = np.mean(capec_cosine_similarities, axis=1)
        capec_mean_cosine_value = round(np.mean(capec_mean_cosine_sim),2)
        capec_mean_cosine_similarity.append(capec_mean_cosine_value)
    
    
    combined_score = [(a+b)/2 for a, b in zip(mean_cosine_similarity,capec_mean_cosine_similarity)]
    max_combined_score = max(combined_score)
    max_combined_index  = combined_score.index(max_combined_score)
    filtered_df = batch_df[batch_df['HMM_label'] == max_combined_index ]
    
    max_mean_cosine_similarity  = max(mean_cosine_similarity)
    max_mean_cosine_index  = mean_cosine_similarity.index(max_mean_cosine_similarity)
    
    max_capec_mean_cosine_similarity  = max(capec_mean_cosine_similarity)
    max_capec_mean_cosine_index = capec_mean_cosine_similarity.index(max_capec_mean_cosine_similarity)
    return batch_df, num_hid_states, filtered_df,max_mean_cosine_similarity,max_mean_cosine_index,mean_cosine_similarity,max_capec_mean_cosine_similarity,\
        max_capec_mean_cosine_index,max_combined_score,capec_mean_cosine_similarity,max_combined_index,combined_score, best_log_likelihood


def HMM_Batch_RecordTimeandMem(batch_df, max_num_state, path, nmf_topic_matrix,
                                mitre_nmf_topic_matrix, capec_nmf_topic_matrix, is_seq=1, num_topics=10):
    print("\n_________HMM learning")
    best_model = None
    best_log_likelihood = float('-inf')
    hmm_time_mem_stats = []  

    for n_states in range(4, max_num_state):
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss
        peak_mem = mem_before  # initialize peak memory as starting memory

        for run in range(10):
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="tied")
            model.fit(nmf_topic_matrix)
            log_likelihood = model.score(nmf_topic_matrix)

            current_mem = process.memory_info().rss
            peak_mem = max(peak_mem, current_mem)

            if log_likelihood > best_log_likelihood:
                best_model = model
                best_log_likelihood = log_likelihood

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        mem_used_mb = round((peak_mem - mem_before) / (1024 * 1024), 2)

        hmm_time_mem_stats.append((n_states, best_log_likelihood, elapsed_time, mem_used_mb))
    
    # Predict hidden states with the best model
    hidden_states = best_model.predict(nmf_topic_matrix)
    batch_df['HMM_label'] = hidden_states
    cluster_count = count_samples_in_clusters(hidden_states)

    num_hid_states = len(np.unique(hidden_states))
    print("\n_________Done HMM. Number of hidden states:", num_hid_states)
    for cluster, count in cluster_count.items():
        print(f"Cluster {cluster}: {count} samples")

    print("\n_________Calculating the similarity score of each hidden state.")    
    mean_cosine_similarity = []
    capec_mean_cosine_similarity = []

    for cluster_id in range(num_hid_states):
        cluster_indices = np.where(hidden_states == cluster_id)[0]
        cluster_vectors = nmf_topic_matrix[cluster_indices]        
        
        # Similarity with MITRE
        cosine_similarities = cosine_similarity(mitre_nmf_topic_matrix, cluster_vectors)
        mean_cosine_value = round(np.mean(np.mean(cosine_similarities, axis=1)), 2)
        mean_cosine_similarity.append(mean_cosine_value)
        
        # Similarity with CAPEC
        capec_cosine_similarities = cosine_similarity(capec_nmf_topic_matrix, cluster_vectors)
        capec_mean_cosine_value = round(np.mean(np.mean(capec_cosine_similarities, axis=1)), 2)
        capec_mean_cosine_similarity.append(capec_mean_cosine_value)

    # Combine and find best cluster
    combined_score = [(a + b) / 2 for a, b in zip(mean_cosine_similarity, capec_mean_cosine_similarity)]
    max_combined_score = max(combined_score)
    max_combined_index = combined_score.index(max_combined_score)
    filtered_df = batch_df[batch_df['HMM_label'] == max_combined_index]

    # Extract max scores
    max_mean_cosine_similarity = max(mean_cosine_similarity)
    max_mean_cosine_index = mean_cosine_similarity.index(max_mean_cosine_similarity)
    max_capec_mean_cosine_similarity = max(capec_mean_cosine_similarity)
    max_capec_mean_cosine_index = capec_mean_cosine_similarity.index(max_capec_mean_cosine_similarity)

    return (batch_df, num_hid_states, filtered_df, 
            max_mean_cosine_similarity, max_mean_cosine_index, mean_cosine_similarity,
            max_capec_mean_cosine_similarity, max_capec_mean_cosine_index,
            max_combined_score, capec_mean_cosine_similarity, max_combined_index, 
            combined_score, best_log_likelihood, hmm_time_mem_stats)
  
def calculate_similarity(cluster_vectors, mitre_nmf_topic_matrix):
    similarities = cosine_similarity(cluster_vectors, mitre_nmf_topic_matrix)
    sim = np.mean(similarities, axis=1)
    return sim

def get_top_indices(similarity_array, original_indices, q):
    sorted_indices = np.argsort(similarity_array)[::-1][:q]
    top_indices = original_indices[sorted_indices]
    return top_indices        

def save_runtime_to_excel(bert_time, bert_mem, nmf_time, nmf_mem, hmm_time_mem_stats=None, filename="runtime_metrics.xlsx"):
    metrics = {
        "Phase": ["BERT Embedding", "NMF Topic Modeling"],
        "Runtime (s)": [bert_time, nmf_time],
        "Memory (MB)": [bert_mem, nmf_mem]
    }
    df_metrics = pd.DataFrame(metrics)

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Sheet 1: Summary
        df_metrics.to_excel(writer, index=False, sheet_name="Runtime Summary")

        # Sheet 2: HMM details
        if hmm_time_mem_stats is not None:
            df_hmm_stats = pd.DataFrame(hmm_time_mem_stats, columns=["# States", "Best Log Likelihood", "Runtime (s)", "Memory (MB)"])
            df_hmm_stats.to_excel(writer, index=False, sheet_name="HMM States Detail")

    print(f"Runtime metrics saved to: {filename}")
#%% Main code

df = pd.read_csv("Data/LMD23_1_75EoRS_Samp2_V5_NovtoJan.csv",encoding="ISO-8859-1")
dataset_name = "LMD23_1_75EoRS_Nov2Jan" 


batch_df = df.copy().reset_index(drop=True)

# =============================================================================
nmf_topic_matrix, batch_df, mitre_nmf_topic_matrix, capec_nmf_topic_matrix, capec_vocabulary, mitre_vocabulary, vocabulary, contextualized_embeddings_pos, \
    mitre_contextualized_embeddings_pos, capec_concat_embeddings_pos, bert_time, bert_mem, nmf_time, nmf_mem = prepare_data(batch_df, path, num_topics = 10)

pic_filename = dataset_name+'_Embs_TopicMatrix.pkl'
with open(pic_filename, 'wb') as f:
    pickle.dump([nmf_topic_matrix, batch_df, mitre_nmf_topic_matrix, capec_nmf_topic_matrix, capec_vocabulary, mitre_vocabulary, vocabulary, contextualized_embeddings_pos, mitre_contextualized_embeddings_pos, capec_concat_embeddings_pos], f)


#%% Perform HMM
max_num_state = 7

batch_df_hmm, num_clusters, filtered_df,max_mean_cosine_similarity,max_mean_cosine_index,mean_cosine_similarity,max_capec_mean_cosine_similarity, max_capec_mean_cosine_index,\
    max_combined_score,capec_mean_cosine_similarity,max_combined_index,combined_score,best_log_likelihood,hmm_time_mem_stats = HMM_Batch_RecordTimeandMem(batch_df, max_num_state, path,nmf_topic_matrix, mitre_nmf_topic_matrix, capec_nmf_topic_matrix, is_seq=1, num_topics=10)

save_hmm_kmeans_results_to_excel_1(
    batch_df, 
    method='HMM', 
    prediction_col_name='HMM_label', 
    dataset_name=dataset_name, 
    num_clusters=num_clusters,
    max_mean_cosine_similarity=max_mean_cosine_similarity,
    max_mean_cosine_index=max_mean_cosine_index,
    mean_cosine_similarity=mean_cosine_similarity,
    max_capec_mean_cosine_similarity=max_capec_mean_cosine_similarity,
    max_capec_mean_cosine_index=max_capec_mean_cosine_index,
    max_combined_score=max_combined_score,
    capec_mean_cosine_similarity=capec_mean_cosine_similarity,
    max_combined_index=max_combined_index,
    combined_score=combined_score,
    score_likehoodOrsilhou=best_log_likelihood,
    nmf_topic_matrix=nmf_topic_matrix,
    mitre_nmf_topic_matrix=mitre_nmf_topic_matrix,
    capec_nmf_topic_matrix=capec_nmf_topic_matrix
)

# After all timing code is completed
save_runtime_to_excel(
    bert_time=bert_time,
    bert_mem=bert_mem,
    nmf_time=nmf_time,
    nmf_mem=nmf_mem,
    hmm_time_mem_stats = hmm_time_mem_stats,
    filename=f"runtime_metrics_{dataset_name}.xlsx"
)    

