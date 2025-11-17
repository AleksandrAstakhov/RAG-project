import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import wikipedia
from langchain_core.documents import Document

from mistralai import Mistral



from mistralai import Mistral
from spellchecker import SpellChecker
from diskcache import Cache

import requests

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-large-latest"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = "./rag_cache"

spell = SpellChecker(language="ru")
cache = Cache(CACHE_DIR)

def load_wikipedia_pages(titles, lang="ru"):
    wikipedia.set_lang(lang)
    docs = []

    for title in titles:
        try:
            print("Загружаю:", title)
            page = wikipedia.page(title, auto_suggest=False)
            text = page.content

            if len(text) < 200:
                print(f"⚠️ Слишком мало текста: {title}")
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": "wikipedia",
                        "title": title,
                        "url": page.url
                    }
                )
            )
            print("Загружено символов:", len(text))

        except Exception as e:
            print(f"Ошибка для {title}: {e}")

    return docs



def split_documents(docs: List[Document], chunk_size=900, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for d in docs:
        chunks = splitter.split_text(d.page_content)
        for i, ch in enumerate(chunks):
            meta = d.metadata.copy() if d.metadata else {}
            meta["chunk"] = i
            out.append(Document(page_content=ch, metadata=meta))
    return out


def build_faiss_index(docs: List[Document]):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    index = FAISS.from_documents(docs, embeddings)
    return index


def make_retriever(index: FAISS, k=4):
    return index.as_retriever(search_type="similarity", search_kwargs={"k": k})


spell = SpellChecker(language="ru")

def fix_typos(text: str) -> str:
    words = text.split()
    misspelled = spell.unknown(words)
    corrected = []
    for w in words:
        if w in misspelled:
            corrected.append(spell.correction(w))
        else:
            corrected.append(w)
    return " ".join(corrected)


def answer_sentence(sentence: str, retriever, llm, ttl=3600):
    corrected = fix_typos(sentence)
    key = f"cache:{corrected}"

    cached = cache.get(key)
    if cached:
        return cached

    docs = retriever.invoke(corrected)

    context = "\n\n---\n\n".join([d.page_content for d in docs])

    prompt = prompt_template.format(sentence=corrected, context=context)
    response = llm(prompt)

    cache.set(key, response, expire=ttl)
    return response


class MistralLLM:
    def __init__(self, api_key, model):
        self.client = Mistral(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        inputs = [
            {"role": "user", "content": prompt}
        ]

        completion_args = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 1
        }

        tools = []

        response = self.client.beta.conversations.start(
            inputs=inputs,
            model="devstral-small-latest",
            instructions="""""",
            completion_args=completion_args,
            tools=tools,
        )
        return response

PROMPT = """Вы — исторический эксперт.
Вам дано предложение, возможно содержащее логические ошибки, неверные даты и факты.
Также в предложении могут быть опечатки.

Используйте данные из контекста (фрагменты из Википедии), чтобы:
исправь неточности, испрваь даты
если какой то факт не верный то удали его

Если всё верно — напишите "Верно" и короткое пояснение.

=== Предложение ===
{sentence}

=== Контекст (фрагменты Википедии) ===
{context}

=== Формат ответа ===
только исправленное предложение
"""

prompt_template = PromptTemplate(
    input_variables=["sentence", "context"],
    template=PROMPT
)

if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        raise ValueError("Установите переменную окружения MISTRAL_API_KEY")
        
    print("Загружаю Википедию...")
    pages = ["Пётр I", "Отечественная война 1812 года", "Великая Отечественная война"]
    raw = load_wikipedia_pages(pages)

    print("Сплитим...")
    chunks = split_documents(raw)

    print(raw)

    print("Строим FAISS...")
    index = build_faiss_index(chunks)


    print("Создаём ретривер...")
    retriever = make_retriever(index, k=4)

    print("LLM = Mistral API...")
    llm = MistralLLM(api_key=MISTRAL_API_KEY, model="magistral-small-latest")

    query = "Пётр Первый родился в 1703 году и участвовал в войне 1812 года."
    result = answer_sentence(query, retriever, llm)

    print("\n=== РЕЗУЛЬТАТ ===")
    print(result.outputs)
