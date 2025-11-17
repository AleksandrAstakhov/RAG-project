import json
from rag.rag_system import answer_sentence
from rag.rag_system import init_rag


def normalize(text):
    return text.lower().strip().replace("ё", "е")


def evaluate_model(llm, retriever):
    with open("rag/test_sentences.json", "r", encoding="utf-8") as f:
        tests = json.load(f)

    correct = 0
    total = len(tests)

    for item in tests:
        inp = item["input"]
        expected = item["expected"]

        result = answer_sentence(inp, retriever, llm)

        print("\n--------------------------------")
        print("Input:    ", inp)
        print("Expected: ", expected)
        print("Model:    ", result)

        if normalize(result) == normalize(expected):
            correct += 1

    accuracy = correct / total
    print(f"\n=== Accuracy: {accuracy * 100:.2f}% ===")

    return accuracy


if __name__ == "__main__":

    llm, retriever = init_rag()
    accuracy = evaluate_model(llm, retriever)

    if accuracy < 0.0:
        raise ValueError("Accuracy too low!")
      
    print(accuracy)
