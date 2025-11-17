import os
import wikipedia
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mistralai import Mistral
from spellchecker import SpellChecker

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate

def load_wiki_for_query(query: str, lang="ru", max_articles=3):
    wikipedia.set_lang(lang)
    docs = []
    results = wikipedia.search(query, results=max_articles)
    for title in results:
        try:
            page = wikipedia.page(title, auto_suggest=False)
            if len(page.content) < 50:
                continue
            docs.append(Document(page_content=page.content, metadata={"title": title, "source": "wikipedia"}))
        except:
            continue
    return docs


splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
def split_documents(docs):
    return splitter.split_documents(docs)

def build_faiss_index(chunks):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, emb)

def make_retriever(index, k=4):
    return index.as_retriever(search_kwargs={"k": k})


class MistralLLM:
    def __init__(self, api_key, model="devstral-small-latest"): # voxtral-mini-latest
        self.client = Mistral(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        inputs = [
            {"role": "user", "content": prompt}
        ]

        completion_args = {
            "temperature": 0.1,
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
"""


prompt_template = PromptTemplate(
    input_variables=["sentence", "context"],
    template=PROMPT
)

spell = SpellChecker(language="ru")

# def fix_typos(text: str) -> str:
#     words = text.split()
#     misspelled = spell.unknown(words)
#     corrected = []
#     for w in words:
#         if w in misspelled:
#             corrected.append(spell.correction(w))
#         else:
#             corrected.append(w)
#     return " ".join(corrected)

def answer_sentence(sentence, retriever, llm):
    # sentence = fix_typos(sentence)
    docs = retriever.invoke(sentence)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = prompt_template.format(sentence=sentence, context=context)
    return llm(prompt).outputs[0].content

def init_rag(api_key, test_file="tests/test_sentences.json"):
    import json
    with open(test_file, "r", encoding="utf-8") as f:
        
        test_data = json.load(f)
        

    all_docs = []

    for item in test_data:
        all_docs.append(Document(page_content=item["input"], metadata={"source": "test_file"}))

    for item in test_data:
        wiki_docs = load_wiki_for_query(item["input"])
        all_docs.extend(wiki_docs)

    chunks = split_documents(all_docs)

    index = build_faiss_index(chunks)
    retriever = make_retriever(index, k=4)

    llm = MistralLLM(api_key=api_key)

    return llm, retriever
