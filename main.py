from src.pipeline import build_pipeline, ask

if __name__ == "__main__":

    retriever, generator = build_pipeline("./data/text_files")

    while True:
        question = input("\nAsk a question or quit:")
        if question.lower() in ["quit","q"]:
            break
        ask(retriever, generator, question)