import os
import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import io
from concurrent.futures import ThreadPoolExecutor

# Oprava pro PyTorch a asyncio
os.environ["PYTORCH_JIT"] = "0"

# Kontrola závislostí
try:
    import sentencepiece
except ImportError:
    st.error("❌ Chybí závislost: `sentencepiece`. Nainstalujte ji pomocí `pip install sentencepiece`.")
    st.stop()

# Nastavení modelů
device = 0  # Pokud máte GPU, nastavte device=0, jinak použijte device=-1 pro CPU
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-cs", device=device)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Použití ThreadPoolExecutor pro paralelní zpracování
executor = ThreadPoolExecutor(max_workers=4)

def extract_text_from_pdf(pdf_file):
    # Extract text from PDF using PyMuPDF
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text_into_chunks(text, max_length=1024):
    # Split text into chunks (based on max_length)
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        if len(" ".join(current_chunk) + " " + word) <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def translate_text(text_chunks):
    # Překlad textu pomocí ThreadPoolExecutor
    translated_text = []
    for chunk in text_chunks:
        future = executor.submit(translator, chunk)
        translated_text.append(future)
    
    results = [future.result() for future in translated_text]
    return [result[0]['translation_text'] for result in results]

def summarize_text(translated_chunks):
    # Sumarizace textu pomocí ThreadPoolExecutor
    summary = []
    for chunk in translated_chunks:
        future = executor.submit(summarizer, chunk, max_length=250, min_length=100, do_sample=False)
        summary.append(future)
    
    results = [future.result() for future in summary]
    return [result[0]["summary_text"] for result in results]

st.title("📄 AI Sumarizátor PDF v češtině")
st.write("Nahrajte PDF soubor a získáte automatický výtah v češtině.")

uploaded_file = st.file_uploader("Nahrajte PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Analyzuji soubor..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        if extracted_text:
            text_chunks = split_text_into_chunks(extracted_text)
            
            # Překlad
            try:
                translated_text = translate_text(text_chunks)
                translated_text = "\n\n".join(translated_text)
            except Exception as e:
                st.error(f"⚠️ Chyba při překladu: {e}")
                st.stop()

            # Sumarizace
            try:
                translated_chunks = split_text_into_chunks(translated_text)
                summary = summarize_text(translated_chunks)
                summary = "\n\n".join(summary)
            except Exception as e:
                st.error(f"⚠️ Chyba při sumarizaci: {e}")
                st.stop()

            with st.expander("📌 Zobrazit výtah v češtině"):
                st.write(summary)

            st.download_button("📥 Stáhnout výtah", summary, file_name="vytah_cz.txt")

            # Tlačítko pro stažení originálního přeloženého textu
            st.download_button("📥 Stáhnout originální CZ text", translated_text, file_name="preklad_cz.txt")
