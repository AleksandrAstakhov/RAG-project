import json
from rag.rag_system import answer_sentence, init_rag
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def is_semantically_close(a, b, threshold=0.85):
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2).item()
    return sim >= threshold, sim


def normalize(text):
    return text.lower().strip().replace("ё", "е")

if __name__ == "__main__":
    import os
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Установите переменную окружения MISTRAL_API_KEY")

    llm, retriever = init_rag(api_key)

    with open("tests/test_sentences.json", "r", encoding="utf-8") as f:
        tests = json.load(f)

    correct = 0
    total = len(tests)

    for item in tests:
        inp = item["input"]
        expected = item["expected"]
        result = answer_sentence(inp, retriever, llm)

        print("\n---")
        print("Input:   ", inp)
        print("Expected:", expected)
        print("Result:  ", result)

        if is_semantically_close(result, expected):
            correct += 1

    accuracy = correct / total
    print(f"\n=== Accuracy: {accuracy*100:.2f}% ===")

    if accuracy < 0.0:
        raise ValueError("Accuracy too low!")
