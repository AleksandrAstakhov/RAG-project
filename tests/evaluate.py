import json
import os
from rag.rag_system import answer_sentence, init_rag, MistralLLM
from langchain_core.prompts import PromptTemplate


CHECK_PROMPT = """
Ты — проверяющий. Имеют ли предложения одинаковый смысл.

Дано:
- Кандидат: "{candidate}"
- Эталон: "{expected}"

Верни СТРОГО Да или Нет.
"""

check_template = PromptTemplate(
    input_variables=["candidate", "expected"], template=CHECK_PROMPT
)


def llm_check(candidate, expected, llm):
    prompt = check_template.format(candidate=candidate, expected=expected)
    out = llm(prompt).outputs[0].content

    try:
        return True if out.lower() == "да" else False
    except:
        return False


if __name__ == "__main__":
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Установите MISTRAL_API_KEY")

    llm_rag, retriever = init_rag(api_key)
    judge_llm = MistralLLM(api_key=api_key)
    llm_no_rag = MistralLLM(api_key=api_key)

    with open("tests/test_sentences.json", "r", encoding="utf-8") as f:
        tests = json.load(f)

    rag_fixed_total = 0
    no_rag_fixed_total = 0

    for item in tests:
        inp = item["input"]
        expected = item["expected"]

        rag_out = answer_sentence(inp, retriever, llm_rag)
        no_rag_out = llm_no_rag(inp).outputs[0].content

        print("\n---")
        print("Input:       ", inp)
        print("Expected:    ", expected)
        print("RAG out:     ", rag_out)
        print("LLM no RAG:  ", no_rag_out)

        rag_fixed = llm_check(rag_out, expected, judge_llm)

        no_rag_fixed = llm_check(no_rag_out, expected, judge_llm)

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
