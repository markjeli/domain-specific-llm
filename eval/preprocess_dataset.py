def doc_to_text(doc) -> str:
    option_choices = {
        "A": doc["ans0"],
        "B": doc["ans1"],
        "C": doc["ans2"],
        "D": doc["ans3"],
    }
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    return f"Question: {doc['question']}\n{answers}Answer:"


def doc_to_target(doc) -> int:
    return doc["label"]