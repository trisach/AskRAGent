import cohere

co = cohere.Client("ob73PqJL0vQ5D7TW2GBAP6PvtmbNQKkg6KTiL7FP")  # get from dashboard
response = co.embed(
    texts=["This is your text"],
    model="embed-english-v3.0",  # or use multilingual
    input_type="search_document"  # required for v3 models
)
embedding = response.embeddings[0]
print(embedding)
