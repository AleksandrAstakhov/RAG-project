import os
import wikipedia
import json
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral
from spellchecker import SpellChecker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from pyaspeller import YandexSpeller
from langchain_community.vectorstores import Chroma

import re


def load_wiki_for_query(query: str, lang="ru", max_articles=3):
    wikipedia.set_lang(lang)
    docs = []
    results = wikipedia.search(query, results=max_articles)
    for title in results:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            if len(page.content) < 50:
                continue
            docs.append(
                Document(
                    page_content=page.content,
                    metadata={"title": title, "source": "wikipedia"},
                )
            )
        except:
            continue
    return docs


splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)


def split_documents(docs):
    return splitter.split_documents(docs)


def build_chroma_index(chunks, persist_dir="chroma_db"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_dir
    )

    db.persist()
    return db


def make_retriever(index, k=4):
    return index.as_retriever(search_kwargs={"k": k})


class MistralLLM:
    def __init__(self, api_key, model="devstral-small-latest"):  # voxtral-mini-latest
        self.client = Mistral(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        inputs = [{"role": "user", "content": prompt}]

        completion_args = {"temperature": 0.1, "max_tokens": 2048, "top_p": 1}

        tools = []
        response = self.client.beta.conversations.start(
            inputs=inputs,
            model=self.model,
            instructions="""""",
            completion_args=completion_args,
            tools=tools,
        )
        return response


PROMPT = """
Вы — эксперт по общим знаниям и популярным фактам.

Ваша задача — исправить ошибки и опечатки в предложении с помощью контекста.
Если факт неверен и вы не знаете правильной версии — удалите только эту часть.

- Сохраняйте исходную логическую полярность высказывания.
  Если предложение было утвердительным — ответ должен быть утвердительным.
  Если предложение было с отрицанием — ответ должен оставаться с отрицанием.
- Нельзя превращать отрицание в утверждение или наоборот.
- Не добавляйте новые факты, которых нет в контексте.
=== Предложение ===
{sentence}

=== Контекст ===
{context}

=== Формат ответа ===
Только исправленное предложение, сохраняя исходную утвердительную/отрицательную форму.
Только исправленное предложение, без добавления новых фактов.
"""


prompt_template = PromptTemplate(
    input_variables=["sentence", "context"], template=PROMPT
)

speller = YandexSpeller(lang="ru")


def fix_typos_yandex(text: str) -> str:
    try:
        fixed_text = speller.spelled(text)
        return fixed_text
    except Exception as e:
        return text


def split_into_sentences(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def answer_sentence(sentence, retriever, llm):
    sentence = fix_typos_yandex(sentence)
    sentences = split_into_sentences(sentence)

    if len(sentences) == 1:
        docs = retriever.invoke(sentence)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = prompt_template.format(sentence=sentence, context=context)
        return llm(prompt).outputs[0].content
    else:
        corrected_sentences = []
        for sent in sentences:
            if len(sent.strip()) > 5:
                docs = retriever.invoke(sent)
                context = "\n\n".join([d.page_content for d in docs])
                prompt = prompt_template.format(sentence=sent, context=context)
                corrected = llm(prompt).outputs[0].content
                corrected_sentences.append(corrected)
            else:
                corrected_sentences.append(sent)

        return ". ".join(corrected_sentences)


def init_rag(api_key, test_file="tests/test_sentences.json"):
    with open(test_file, "r", encoding="utf-8") as f:

        test_data = json.load(f)

    all_docs = []

    for item in test_data:
        all_docs.append(
            Document(page_content=item["input"], metadata={"source": "test_file"})
        )

    for item in test_data:
        wiki_docs = load_wiki_for_query(item["input"])
        all_docs.extend(wiki_docs)

    chunks = split_documents(all_docs)

    index = build_chroma_index(chunks)
    retriever = make_retriever(index, k=4)

    llm = MistralLLM(api_key=api_key)

    return llm, retriever
