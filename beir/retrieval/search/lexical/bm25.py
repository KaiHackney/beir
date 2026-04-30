import math
from collections import Counter

class BM25:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_freqs = self._calculate_doc_freqs()
        self.N = len(corpus)

    def _calculate_doc_freqs(self):
        doc_freqs = Counter()
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freqs[term] += 1
        return doc_freqs

    def idf(self, term):
        n_t = self.doc_freqs.get(term, 0)
        return math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query, doc):
        score = 0
        doc_term_freqs = Counter(doc)
        for term in query:
            f_t_D = doc_term_freqs[term]
            idf_t = self.idf(term)
            numerator = f_t_D * (self.k1 + 1)
            denominator = f_t_D + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
            score += idf_t * (numerator / denominator)
        return score

    def search(self, queries, top_k=10):
        results = []
        for query in queries:
            scores = [(i, self.score(query, doc)) for i, doc in enumerate(self.corpus)]
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
            results.append(scores)
        return results