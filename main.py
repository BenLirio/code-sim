from openai import Embedding
from openai.embeddings_utils import cosine_similarity

def get_embedding(code):
    response = Embedding.create(
        input=code,
        model="code-search-babbage-code-001"
    )
    embeddings = response['data'][0]['embedding']
    return embeddings


funcA = """int a = 3;
int b = 2;
int c = a + b;"""

funcB = """int b = 2;
int a = 3;
int c = a + b;"""

funcC = """int c = 2 + 3"""

embeddingA = get_embedding(funcA)
embeddingB = get_embedding(funcB)
embeddingC = get_embedding(funcC)

simAB = cosine_similarity(embeddingA, embeddingB)
simAC = cosine_similarity(embeddingA, embeddingC)
print(f"SimAB: {simAB}")
print(f"SimAC: {simAC}")
