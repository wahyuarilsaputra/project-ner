import pandas as pd
import numpy as np
# from gensim.models import Word2Vec
# from tensorflow.keras.utils import pad_sequences
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from pypdf import PdfReader
import re

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["word"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]

        # sentences_words = [[w[0] for w in s] for s in self.sentences]
        # self.cbow_model = Word2Vec(sentences_words, vector_size=10, window=5, min_count=1, sg=0, seed=42)

    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None

def process_data_and_train_model(file_path):
    # Load and process data
    data = pd.read_csv(file_path, encoding="latin1")
    data['word'] = data['word'].replace(to_replace ='\d+', value = '<\g<0>>', regex = True)
    data = data.fillna(method="ffill")
    data['tag'] = data['tag'].apply(lambda x: [x])

    # Combine tags in the same sentence and remove sentences with only 'O' tags
    df_combined = (
        data.groupby(['doc', 'sentence'])['tag']
        .agg(lambda x: sum(x, []))
        .apply(lambda tags: tags if any(tag != 'O' for tag in tags) else [])
        .reset_index(name='tag')
    )

    # Remove rows with no tags or empty tags
    df_combined = df_combined[df_combined['tag'].apply(lambda x: bool(x))]

    # Create a new DataFrame containing rows that match df_combined
    filtered_df = data[data['sentence'].isin(df_combined['sentence'])]

    # Separate df_combined by tag and fill in sentence based on the separated tags
    result_rows = []
    for index, row in filtered_df.iterrows():
        for tag in row['tag']:
            result_rows.append({
                'doc': row['doc'],
                'sentence': row['sentence'],
                'word': row['word'],
                'prev': row['prev'],
                'next': row['next'],
                'tag': tag
            })

    # Display the dataframe after merging, deleting, separating back tags, and filling in the 'sentence' column
    data = pd.DataFrame(result_rows)

    words = list(set(data["word"].values))
    words.append("ENDPAD")
    n_words = len(words)
    tags = list(set(data["tag"].values))
    n_tags = len(tags)

    # Initialize SentenceGetter and retrieve sentences
    getter = SentenceGetter(data)
    sentences = getter.sentences
    sentences_words = [[w[0] for w in s] for s in getter.sentences]

    # Train Word2Vec model
    cbow_model = Word2Vec(sentences_words, vector_size=10, window=5, min_count=1, sg=0, seed=42)
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    MAX_LEN = 200

    def get_word_embedding(word):
        if word in cbow_model.wv:
            return cbow_model.wv[word]
        else:
            return np.zeros(cbow_model.vector_size)

    X = [[get_word_embedding(w[0]) for w in s] for s in sentences]
    X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", dtype='float32', value=np.zeros(cbow_model.vector_size))

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=n_tags+1) for i in y]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False,random_state=42)
    X_tr.shape, X_te.shape, np.array(y_tr).shape, np.array(y_te).shape

    return data, cbow_model, MAX_LEN

def pdf_to_text(file_path):
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.replace("Mahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nMahkamah Agung Republik Indonesia\nDirektori Putusan Mahkamah Agung Republik Indonesia\nputusan.mahkamahagung.go.id\n", "")
    text = text.replace("\nDisclaimer\nKepaniteraan Mahkamah Agung Republik Indonesia berusaha untuk selalu mencantumkan informasi paling kini dan akurat sebagai bentuk komitmen Mahkamah Agung untuk pelayanan publik, transparansi dan akuntabilitas\npelaksanaan fungsi peradilan. Namun dalam hal-hal tertentu masih dimungkinkan terjadi permasalahan teknis terkait dengan akurasi dan keterkinian informasi yang kami sajikan, hal mana akan terus kami perbaiki dari waktu kewaktu.\nDalam hal Anda menemukan inakurasi informasi yang termuat pada situs ini atau informasi yang seharusnya ada, namun belum tersedia, maka harap segera hubungi Kepaniteraan Mahkamah Agung RI melalui :\nEmail : kepaniteraan@mahkamahagung.go.id", "")
    text = text.replace('P U T U S A N', 'PUTUSAN').replace('T erdakwa', 'Terdakwa').replace('T empat', 'Tempat').replace('T ahun', 'Tahun')
    text = text.replace('P  E  N  E  T  A  P  A  N', 'PENETAPAN').replace('J u m l a h', 'Jumlah').replace('M E N G A D I L I', 'MENGADILI')
    text = re.sub(r'(Hal\.\s*\S+(?:\s*\S+)?\.?\s*)|(Halaman \d+(?:\.\d+)?\s*)|Putusan Nomor \S+\s*', '', text)
    text = re.sub(r'\b0+(\d+)', r'\1', text)
    text = text.replace('\uf0d8', '').replace('\uf0b7', '').replace('\n', ' ')
    text = re.sub(r'([“”"])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'halaman\s*\d+\s*dari\s*\d+\s*', '', text)
    text = re.sub(r'^\s*dari\s+\d+\s+bkl\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*dari\s+\d+\s+smp\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+/PN', '/PN', text)
    text = re.sub(r'(\d+)\s*/pid', r'\1/pid', text)
    text = re.sub(r'(?i)(nomor)(\d+)', r'\1 \2', text)
    text = re.sub(r'(\d+)/\s*(pid\.\w+)/\s*(\d{4})/\s*(pn)', r'\1/\2/\3/\4', text, flags=re.IGNORECASE)
    return text.lower().strip()

def multiple_replace(text, replacements):
    regex = re.compile('|'.join(re.escape(key) for key in replacements.keys()))

    def replace(match):
        return replacements[match.group(0)]

    return regex.sub(replace, text)

def tokenize_text(text):
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens

def get_word_embedding(word, cbow_model):
    if word in cbow_model.wv:
        return cbow_model.wv[word]
    else:
        return np.zeros(cbow_model.vector_size)

def predict_labels(model, sentence, target_tags, cbow_model, idx2tag):
    MAX_LEN = 200
    sentences = sentence.split(';')
    combined_results = {key: "" for key in target_tags}
    found_terdakwa = []
    tags_from_start = ["nomor putusan","nama terdakwa",  "melanggar UU", "putusan hukuman", "tuntutan hukuman"]
    tags_from_end = ["jenis dakwaan","tanggal putusan", "hakim ketua", "hakim anggota", "penuntut umum", "panitera"]

    for sent in sentences:
        words = sent.strip().split()
        words = [re.split(r'(\W)', word) for word in words]
        words = [item for sublist in words for item in sublist if item.strip()]
        embeddings = [get_word_embedding(word, cbow_model) for word in words]
        padded_sequence = pad_sequences([embeddings], maxlen=MAX_LEN, padding="post", dtype='float32', value=np.zeros(cbow_model.vector_size))
        predictions = model.predict(padded_sequence)
        predicted_labels = np.argmax(predictions, axis=-1)[0]
        predicted_tags = [idx2tag[idx] for idx in predicted_labels]
        temp_results = {key: [] for key in target_tags}

        for word, tag in zip(words, predicted_tags):
            for key, tags in target_tags.items():
                if tag in tags:
                    temp_results[key].append(word)
                    
        for key in temp_results:
            if key in tags_from_start and not combined_results[key] and temp_results[key]:
                combined_results[key] = ' '.join(sorted(set(temp_results[key]), key=temp_results[key].index))

    for sent in reversed(sentences):
        if all(combined_results[key] for key in tags_from_end):
            break
        words = sent.strip().split()
        words = [re.split(r'(\W)', word) for word in words]
        words = [item for sublist in words for item in sublist if item.strip()]
        embeddings = [get_word_embedding(word, cbow_model) for word in words]
        padded_sequence = pad_sequences([embeddings], maxlen=MAX_LEN, padding="post", dtype='float32', value=np.zeros(cbow_model.vector_size))
        predictions = model.predict(padded_sequence)
        predicted_labels = np.argmax(predictions, axis=-1)[0]
        predicted_tags = [idx2tag[idx] for idx in predicted_labels]
        temp_results = {key: [] for key in target_tags}

        for word, tag in zip(words, predicted_tags):
            for key, tags in target_tags.items():
                if tag in tags:
                    temp_results[key].append(word)

        for key in temp_results:
            if key in tags_from_end and not combined_results[key] and temp_results[key]:
                combined_results[key] = ' '.join(sorted(set(temp_results[key]), key=temp_results[key].index))

    if combined_results['hakim anggota']:
        combined_results['hakim anggota'] = process_hakim_anggota(combined_results['hakim anggota'])
                
    return combined_results

def process_hakim_anggota(text):
    matches = re.findall(r'(\w+\s\w+)', text)
    if matches:
        return ' dan '.join(matches)
    return text
