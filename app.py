import os
import asyncio
import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import io

# Oprava pro PyTorch a asyncio
os.environ["PYTORCH_JIT"] = "0"
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Kontrola závislostí
try:
    import sentencepiece
except ImportError:
    st.error("❌ Chybí závislost: `sentencepiece`. Nainstalujte ji pomocí `pip install sentencepiece`.")
    st.stop()

# Nastavení modelů
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-cs")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        # Převedení souboru do čitelného formátu
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"❌ Chyba při čtení PDF: {e}")
        return None
    return text

def split_text_into_chunks(text, max_length=1024):
    sentences = text.split(". ")
    chunks = []
    chunk = ""
    
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    
    if chunk:
        chunks.append(chunk.strip())
    
    return chunks

st.title("📄 AI Sumarizátor PDF v češtině")
st.write("Nahrajte PDF soubor a získáte automatický výtah v češtině.")

uploaded_file = st.file_uploader("Nahrajte PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("🔍 Analyzuji soubor..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        if extracted_text:
            text_chunks = split_text_into_chunks(extracted_text)
            
            translated_text = ""
            for chunk in text_chunks:
                try:
                    translated_part = translator(chunk)[0]['translation_text']
                    translated_text += translated_part + "\n\n"
                except Exception as e:
                    st.error(f"⚠️ Chyba při překladu: {e}")

            translated_chunks = split_text_into_chunks(translated_text)
            summary = ""
            for chunk in translated_chunks:
                try:
                    summary_part = summarizer(chunk, max_length=250, min_length=100, do_sample=False)[0]["summary_text"]
                    summary += summary_part + "\n\n"
                except Exception as e:
                    st.error(f"⚠️ Chyba při sumarizaci: {e}")

            with st.expander("📌 Zobrazit výtah v češtině"):
                st.write(summary)

            st.download_button("📥 Stáhnout výtah", summary, file_name="vytah_cz.txt")
