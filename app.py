import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import gensim
from gensim.models import Word2Vec
from processing import process_data_and_train_model, pdf_to_text, preprocess_text, multiple_replace, tokenize_text, get_word_embedding, predict_labels, process_hakim_anggota
from tensorflow.keras.models import load_model

def main():
    st.sidebar.title("NER pada dokumen putusan pengadilan")
    menu = ["Implementasi", "Penjelasan Program"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Implementasi":
        st.title("Named Entity Recognition pada dokumen putusan pengadilan")
        st.markdown("<h4>Masukan file dokumen</h4>", unsafe_allow_html=True)
        input_pdf = st.file_uploader(
            "Input PDF",
            label_visibility="hidden",
            type=["pdf"],
            key="pdf_input",
        )
        model_choice = st.selectbox("Pilih Optimizer", ["SGD", "Adam", "RMSprop"])
        btn_process = st.button("Proses Dokumen", type="primary" )
        if btn_process:
            if input_pdf:
                with st.spinner("Memproses file PDF..."):
                    text = pdf_to_text(input_pdf)
                    preprocessed_text = preprocess_text(text)
                    replacements = {
                        '1 0': '10' ,'20 20': '2020', '2 7' : '27', 'a gustus' : 'agustus', '( 2)': '(2)', '.,': ' .,', ',s': ' ,s','ted dy' : 'teddy',
                        'ro omius,' : 'roomius,', 'h ukum': 'hukum', '//pn': '/pn', '2022oleh' : '2022 oleh', 'alimuddi n,' : 'alimuddin,', '202 3': '2023', 
                        'perkos aan': 'perkosaan', 'memberatka n': 'memberatkan', 'p idana': 'pidana', 'sebag aimana': 'sebagaimana', 'bulan;' : 'bulan', '2 23' : '2023',
                        '2 023': '2023','2 022': '2022', 'f isik': 'fisik', 'be rsalah': 'bersalah', '(sepuluh )': '(sepuluh)', 'ap ril' : 'april',
                        '(enam )': '(enam)', 'no mor': 'nomor', '( 3)': '(3)', 'an ak': 'anak', 'tuj uh': 'tujuh', 'hu bungan': 'hubungan', 'pencur ian' : 'pencurian',
                        'selam a1 (satu)' : 'selama 1 (satu)', '(empat )' : '(empat)', 's eptember' : 'september', 'keadaa n' : 'keadaan', '27januari' : '27 januari',
                        'bulan;3.menetapkan' : 'bulan 3.menetapkan','januari,2023' : 'januari 2023', '202 2' : '2022','202 4' : '2024', '202 3' : '2023','2023//' : '2023', 
                        '20 22' : '2022','20 21' : '2021','202 1' : '2021', '27januari' : '27 januari', 'januari,2023' : 'januari 2023', 'sepember' : 'september',
                        'bk l' : 'bkl','pm k' : 'pmk', 'bki' : 'bkl',  'pid.b /' : 'pid.b/', 'pn.' : 'pn ', 'pid. b' : 'pid.b', 'pnpmk' : 'pn pmk', '2023pn' : '2023 pn',
                        'smp.' : 'smp', '(al m)' : '(alm)', 'Al m' : 'Alm', "'" : '’', 'b in' : 'bin', 'bu lan' : 'bulan', 'kuhpdan' : 'kuhp dan',
                        'pencuriandalam' : 'pencurian dalam', 'keadaa n' : 'keadaan', 'pembunuhanberencana' : 'pembunuhan berencana', 'bulan3.menetapkan' : 'bulan 3.menetapkan', 
                        'bulan3.menyatakan' :  'bulan 3.menyatakan', 't aufik' : 'taufik', 'olehhaidir' : 'oleh haidir','olehfajrini' : 'oleh fajrini', 'janurai' : 'januari',
                        'nopember': 'november','pebruari': 'februari','a gustus' : 'agustus', 'september2023,' : 'september 2023,','aapril' : 'april', 'ja nuari' : 'januari',
                        'no mor': 'nomor', ',serta' : ', serta', 'agussyamsul' : 'agus syamsul', 'a stuti' : 'astuti'
                    }
                    preprocessed_text = multiple_replace(preprocessed_text, replacements)
                    st.subheader("Text PDF")
                    st.text(preprocessed_text)
                    tokens = tokenize_text(preprocessed_text)
                    if model_choice == "SGD":
                        save_path = 'Modeling\Hasil Training\ModelSGD.keras'
                        model = load_model(save_path)
                        model.load_weights('Modeling\Hasil Training\modelSGD.weights.h5')
                    elif model_choice == "Adam":
                        save_path = 'Modeling\Hasil Training\modelAdam.keras'
                        model = load_model(save_path)
                        model.load_weights('Modeling\Hasil Training\modelAdam.weights.h5')
                    elif model_choice == "RMSprop":
                        save_path = 'Modeling\Hasil Training\modelRmsprop.keras'
                        model = load_model(save_path)
                        model.load_weights('Modeling\Hasil Training\modelRmsprop.weights.h5')
                    cbow_model = Word2Vec.load("Modeling\Model_CBOW\CBOWModel.model")
                    idx2tag = {0: 'B_ARTV', 1: 'B_CRIA', 2: 'B_DEFN', 3: 'B_JUDP', 4: 'B_JUG', 5: 'B_PENA', 6: 'B_PROS', 7: 'B_PUNI', 8: 'B_REGI', 9: 'B_TIMV', 10: 'B_VERN',
                            11: 'I_ARTV', 12: 'I_CRIA', 13: 'I_DEFN', 14: 'I_JUDP', 15: 'I_JUG', 16: 'I_PENA', 17: 'I_PROS', 18: 'I_PUNI', 19: 'I_REGI', 20: 'I_TIMV',
                            21: 'I_VERN', 22: 'O'}
                    target_tags = {"nomor putusan": ["B_VERN", "I_VERN"], "nama terdakwa": ["B_DEFN", "I_DEFN"], "jenis dakwaan": ["B_CRIA", "I_CRIA"],
                    "melanggar UU": ["B_ARTV", "I_ARTV"], "putusan hukuman": ["B_PUNI", "I_PUNI"], "tuntutan hukuman": ["B_PENA", "I_PENA"],
                    "tanggal putusan": ["B_TIMV", "I_TIMV"], "hakim ketua": ["B_JUDP", "I_JUDP"], "hakim anggota": ["B_JUG", "I_JUG"],
                    "penuntut umum": ["B_PROS", "I_PROS"], "panitera": ["B_REGI", "I_REGI"]}
                    relevant_words = predict_labels(model, preprocessed_text, target_tags, cbow_model, idx2tag)
                    st.subheader("Hasil Prediksi:")
                    dataOutput = []
                    for key, words in relevant_words.items():
                        if words:
                            if key.lower() == 'nomor putusan':
                                output = words.replace(' / ', '/').replace(' ','').replace('pn','pn ').replace('.b','.b/').replace('.c','.c/').replace('2019','2019/').replace('2020','2020/').replace('2021','2021/').replace('2022','2022/').replace('2023','2023/').replace('2024','2024/')
                            else:
                                output = words.replace('“','').replace('”','')
                        else:
                            output = f"Tidak ada kata dengan tag {key} dalam kalimat ini."
                        dataOutput.append([key.capitalize(), output])
                    
                    df = pd.DataFrame(dataOutput, columns=["Tag", "Output"])
                    st.table(df)
    elif choice == "Penjelasan Program":
        st.title("Named Entity Recognition pada dokumen putusan pengadilan")
        st.markdown("<h4>Penjelasan Program</h4>", unsafe_allow_html=True)
        st.write("""
        <div style="text-align: justify; text-indent: 2em;">
            <p>Penelitian mengenai implementasi Named Entity Recognition (NER) dilakukan untuk melakukan identifikasi dan ekstraksi informasi pada dokumen putusan pengadilan yang berfokus menggunakan dataset yang diambil dari pengadilan negeri di Pulau Madura. Penelitian ini menggunakan arsitektur Bi-LSTM (Bidirectional Long Short-Term Memory).</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
