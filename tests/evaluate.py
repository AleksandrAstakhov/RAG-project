import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.rag_system import answer_sentence, init_rag, MistralLLM


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

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb_yes = embed_model.encode("Да", convert_to_numpy=True)
emb_no = embed_model.encode("Нет", convert_to_numpy=True)


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def llm_check_embed(answer: str) -> bool:
    """True → ближе к 'Да', False → ближе к 'Нет'."""
    emb_ans = embed_model.encode(answer, convert_to_numpy=True)
    return cos_sim(emb_ans, emb_yes) >= cos_sim(emb_ans, emb_no)


if __name__ == "__main__":
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Установите MISTRAL_API_KEY")

    llm_rag, retriever = init_rag(api_key)

    llm_no_rag = MistralLLM(api_key=api_key, model="voxtral-small-latest")

    with open("tests/test_sentences.json", "r", encoding="utf-8") as f:
        tests = json.load(f)

    rag_fixed_total = 0
    no_rag_fixed_total = 0

    for item in tests:
        inp = item["input"]
        expected = item["expected"]

        rag_out = answer_sentence(inp, retriever, llm_rag)

        no_rag_prompt = PROMPT.format(sentence=inp, context="")
        no_rag_resp = llm_no_rag(no_rag_prompt)

        if hasattr(no_rag_resp, "content"):
            no_rag_out = no_rag_resp.content
        else:
            no_rag_out = no_rag_resp.outputs[0].content

        print("\n---")
        print("Input:       ", inp)
        print("Expected:    ", expected)
        print("RAG out:     ", rag_out)
        print("LLM no RAG:  ", no_rag_out)

        rag_fixed = llm_check_embed(rag_out)
        no_rag_fixed = llm_check_embed(no_rag_out)

        print("RAG fixed?    ", rag_fixed)
        print("NO_RAG fixed? ", no_rag_fixed)

        if rag_fixed:
            rag_fixed_total += 1
        if no_rag_fixed:
            no_rag_fixed_total += 1

    n = len(tests)

    print("\n===== ИТОГ =====")
    print(f"RAG исправил ошибок:     {rag_fixed_total} / {n}")
    print(f"NO_RAG исправил ошибок:  {no_rag_fixed_total} / {n}")

    if rag_fixed_total == 0:
        raise ValueError("RAG плохо исправляет ошибки!")
