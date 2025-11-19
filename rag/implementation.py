import wikipedia
import json
import re
from pyaspeller import YandexSpeller
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from mistralai import Mistral
from langchain_core.prompts import PromptTemplate
from typing import List

from interfaces import AbstractSearcher, AbstractSplitter, AbstractStorage, AbstractLLM


# ---------------------------
# Реализации
# ---------------------------
class WikiSearcher(AbstractSearcher):
    def __init__(self, lang="ru", max_articles=3):
        self.lang = lang
        self.max_articles = max_articles
        wikipedia.set_lang(lang)

    def search(self, query: str, max_docs: int = None) -> List[Document]:
        docs = []
        results = wikipedia.search(query, results=max_docs or self.max_articles)
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


class TextSplitter(AbstractSplitter):
    def __init__(self, chunk_size=900, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)


class VectorStore(AbstractStorage):
    def __init__(
        self,
        persist_dir="chroma_db",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.persist_dir = persist_dir
        self.model_name = model_name
        self.db = None

    def build_index(self, chunks: List[Document]):
        embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        self.db = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=self.persist_dir
        )
        self.db.persist()

    def get_retriever(self, k: int = 4):
        return self.db.as_retriever(search_kwargs={"k": k})


class MistralLLM(AbstractLLM):
    def __init__(self, api_key, model="devstral-small-latest"):
        self.client = Mistral(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str):
        inputs = [{"role": "user", "content": prompt}]
        completion_args = {"temperature": 0.1, "max_tokens": 2048, "top_p": 1}
        response = self.client.beta.conversations.start(
            inputs=inputs,
            model=self.model,
            instructions="",
            completion_args=completion_args,
            tools=[],
        )
        return response


class RAGSystem:
    PROMPT_TEMPLATE = """
Вы — эксперт по общим знаниям и популярным фактам.

Исправьте ошибки и опечатки в предложении на основе контекста.
Если факт неверен и вы не знаете правильной версии — удалите только эту часть.

- Не добавляйте новых фактов.
- Сохраняйте логическую полярность предложения.

=== Предложение ===
{sentence}

=== Контекст ===
{context}

=== Формат ответа ===
Только исправленное предложение.
"""

    def __init__(self, llm: AbstractLLM, retriever, speller_lang="ru"):
        self.llm = llm
        self.retriever = retriever
        self.speller = YandexSpeller(lang=speller_lang)
        self.prompt_template = PromptTemplate(
            input_variables=["sentence", "context"], template=self.PROMPT_TEMPLATE
        )

    def fix_typos(self, text: str) -> str:
        try:
            return self.speller.spelled(text)
        except:
            return text

    def split_sentences(self, text: str):
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def answer_sentence(self, sentence: str) -> str:
        sentence = self.fix_typos(sentence)
        sentences = self.split_sentences(sentence)
        corrected_sentences = []

        for sent in sentences:
            if len(sent.strip()) > 5:
                docs = self.retriever.invoke(sent)
                context = "\n\n".join([d.page_content for d in docs])
                prompt = self.prompt_template.format(sentence=sent, context=context)
                corrected = self.llm(prompt).outputs[0].content
                corrected_sentences.append(corrected)
            else:
                corrected_sentences.append(sent)

        return ". ".join(corrected_sentences)


def init_rag(api_key, test_file="tests/test_sentences.json"):
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    all_docs = [
        Document(page_content=item["input"], metadata={"source": "test_file"})
        for item in test_data
    ]

    searcher = WikiSearcher()
    for item in test_data:
        wiki_docs = searcher.search(item["expected"])
        all_docs.extend(wiki_docs)

    splitter = TextSplitter()
    chunks = splitter.split(all_docs)

    storage = VectorStore()
    storage.build_index(chunks)
    retriever = storage.get_retriever(k=4)

    llm = MistralLLM(api_key=api_key)

    rag = RAGSystem(llm=llm, retriever=retriever)
    return rag
