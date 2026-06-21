from openai import OpenAI
from pathlib import Path

BASE_DIR = Path(__file__).parent

client = OpenAI()

PDF_PATH = BASE_DIR / "files" / "Teknologie masterexamen Haoyang Zhang.pdf"
ID_FILE = BASE_DIR / "vector_store_id.txt"

def get_or_create_vector_store():
    if ID_FILE.exists():
        vector_store_id = ID_FILE.read_text().strip()
        print(f"Using existing vector store: {vector_store_id}")
        return vector_store_id
    
    vector_store = client.vector_stores.create(
        name="Thesis resolve"
    )

    with open(PDF_PATH, "rb") as f:
        client.vector_stores.files.upload_and_poll(
            vector_store_id=vector_store.id,
            file=f,
    )

    ID_FILE.write_text(vector_store.id)

    print(f"Created new vector store: {vector_store.id}")

    return vector_store.id



def format_sources(results):
    sources = ""

    for i, result in enumerate(results.data,start = 1):
        sources += f"\n[source {i}: {result.filename}, score={result.score}]\n"

        for chunk in result.content:
            sources += chunk.text + "\n"

    return sources


def ask_pdf(question: str, vector_store_id: str):
    results = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=question,
        max_num_results = 3,
    )

    sources = format_sources(results)

    response = client.responses.create(
        model="gpt-5.5",
        instructions=(
            "You are a helpful document QA assistant. "
            "Answer the user's question using only the provided sources. "
            "If the answer is not in the sources, say you don't know."
        ),
        input=f"""
Sources:
{sources}

Question:
{question}
"""
    )

    return response.output_text


def search_pdf(question: str):
    vector_store_id = get_or_create_vector_store()
    return ask_pdf(question, vector_store_id)

def main():
    VECTOR_STORE_ID = get_or_create_vector_store()

    while True:
        question = input("\n请输入问题（输入 exit 退出）：")

        if question.lower() == "exit":
            break

        answer = ask_pdf(question,VECTOR_STORE_ID)
        print("\n回答：")
        print(answer)

if __name__ == "__main__":
    main()