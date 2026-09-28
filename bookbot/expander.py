SYNONYMS = {
    "nn": ["neural network", "neural net"],
    "ml": ["machine learning"],
    "dl": ["deep learning"],
    "ai": ["artificial intelligence"],
    "xss": ["cross-site scripting"],
    "rce": ["remote code execution"],
    "privesc": ["privilege escalation"],
    "cve": ["vulnerability"],
    "cnn": ["convolutional neural network"],
    "rnn": ["recurrent neural network"],
    "lstm": ["long short-term memory"],
}


def expand(query):
    tokens = query.lower().split()
    extras = []
    for t in tokens:
        if t in SYNONYMS:
            extras.extend(SYNONYMS[t])
    if extras:
        return query + " " + " ".join(extras)
    return query