"""Comprehensive text-to-numbers vectorization suite for Oxideon.

Provides classic statistical vectorizers (CountVectorizer, TfidfTransformer, TfidfVectorizer,
HashingVectorizer) and advanced semantic/neural vectorizers (BM25Vectorizer, Word2VecVectorizer,
PositionalSpectralVectorizer, ContextualPoolingVectorizer).
"""

from collections import Counter
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from .oxideon import (
    Tensor as _Tensor,
    from_numpy as _from_numpy,
    BpeTokenizer,
    FastBM25Index,
    FastCountVectorizer,
    FastTfidfVectorizer,
    FastHashingVectorizer,
    FastTextCharNGramVectorizer,
    FastMinHashVectorizer,
    FastSimHashVectorizer,
    FastByteHistogramVectorizer,
    FastMaxSimScorer,
    FastSphericalHarmonics,
    FastDiscretizer,
    FastLsaVectorizer,
    FastDictVectorizer,
    FastOneHotVectorizer,
    FastJsonTreeVectorizer,
    FastTimeSeriesSpectralVectorizer,
    FastPathSignatureVectorizer,
    FastDeepSetsVectorizer,
    FastWeisfeilerLehmanVectorizer,
    KernelSpectralVectorizer,
    fast_murmurhash3_32,
)
from .sparse import CSRTensor, COOTensor


# Standard English Stop Words (scikit-learn / NLTK standard union)
ENGLISH_STOP_WORDS: frozenset[str] = frozenset({
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "all",
    "almost", "alone", "along", "already", "also", "although", "always", "am", "among",
    "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone",
    "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became",
    "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind",
    "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom",
    "but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry",
    "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
    "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever",
    "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fifty",
    "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found",
    "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt",
    "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon",
    "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if",
    "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last",
    "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile",
    "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must",
    "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine",
    "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off",
    "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise",
    "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please",
    "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious",
    "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so",
    "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still",
    "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves",
    "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon",
    "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through",
    "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via",
    "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where",
    "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether",
    "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
    "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves",
})


def murmurhash3_32(key: str, seed: int = 0) -> int:
    """Deterministic, high-speed pure-Python MurmurHash3 32-bit implementation."""
    data = key.encode("utf-8")
    length = len(data)
    nblocks = length // 4
    h1 = seed & 0xFFFFFFFF
    c1 = 0xCC9E2D51
    c2 = 0x1B873593

    for i in range(0, nblocks * 4, 4):
        k1 = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16) | (data[i + 3] << 24)
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF

    tail_index = nblocks * 4
    k1 = 0
    tail_len = length & 3
    if tail_len == 3:
        k1 ^= data[tail_index + 2] << 16
    if tail_len >= 2:
        k1 ^= data[tail_index + 1] << 8
    if tail_len >= 1:
        k1 ^= data[tail_index]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    h1 ^= length
    h1 ^= (h1 >> 16)
    h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
    h1 ^= (h1 >> 13)
    h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
    h1 ^= (h1 >> 16)
    return h1


class CountVectorizer:
    """Convert a collection of text documents to a matrix of token counts.

    Produces either an Oxideon Tensor or a Sparse CSRTensor.
    """

    def __init__(
        self,
        lowercase: bool = True,
        stop_words: Optional[Union[str, Iterable[str], Set[str]]] = None,
        token_pattern: str = r"(?u)\b\w\w+\b",
        ngram_range: Tuple[int, int] = (1, 1),
        max_df: Union[float, int] = 1.0,
        min_df: Union[float, int] = 1,
        max_features: Optional[int] = None,
        binary: bool = False,
    ):
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.binary = binary

        self.vocabulary_: Dict[str, int] = {}
        self.stop_words_: Set[str] = set()
        self._compiled_regex = re.compile(token_pattern)

    def _resolve_stop_words(self) -> Set[str]:
        if self.stop_words is None:
            return set()
        if isinstance(self.stop_words, str):
            if self.stop_words == "english":
                return set(ENGLISH_STOP_WORDS)
            raise ValueError(f"Unknown stop words specification '{self.stop_words}'")
        return set(self.stop_words)

    def build_tokenizer(self) -> Callable[[str], List[str]]:
        regex = self._compiled_regex
        return lambda doc: regex.findall(doc)

    def build_analyzer(self) -> Callable[[str], List[str]]:
        tokenizer = self.build_tokenizer()
        stop_set = self._resolve_stop_words()
        min_n, max_n = self.ngram_range

        def _analyze(doc: str) -> List[str]:
            if self.lowercase:
                doc = doc.lower()
            tokens = tokenizer(doc)
            if stop_set:
                tokens = [t for t in tokens if t not in stop_set]

            if min_n == 1 and max_n == 1:
                return tokens

            ngrams = []
            n_tokens = len(tokens)
            for n in range(min_n, max_n + 1):
                for i in range(n_tokens - n + 1):
                    ngrams.append(" ".join(tokens[i : i + n]))
            return ngrams

        return _analyze

    def fit(self, raw_documents: Iterable[str]) -> "CountVectorizer":
        """Learn a vocabulary dictionary of all tokens in the raw documents."""
        analyzer = self.build_analyzer()
        term_doc_counts: Counter[str] = Counter()
        term_freqs: Counter[str] = Counter()
        n_docs = 0

        for doc in raw_documents:
            n_docs += 1
            terms = analyzer(doc)
            term_freqs.update(terms)
            term_doc_counts.update(set(terms))

        if n_docs == 0:
            self.vocabulary_ = {}
            return self

        max_doc_count = self.max_df if isinstance(self.max_df, int) else int(self.max_df * n_docs)
        min_doc_count = self.min_df if isinstance(self.min_df, int) else int(self.min_df * n_docs)

        # Filter by document frequencies
        candidates = [
            term
            for term, doc_cnt in term_doc_counts.items()
            if min_doc_count <= doc_cnt <= max_doc_count
        ]

        # Filter by max_features (highest term frequency)
        if self.max_features is not None and len(candidates) > self.max_features:
            candidates.sort(key=lambda t: (-term_freqs[t], t))
            candidates = candidates[: self.max_features]

        candidates.sort()
        self.vocabulary_ = {term: idx for idx, term in enumerate(candidates)}
        self.stop_words_ = self._resolve_stop_words()
        return self

    def transform(
        self,
        raw_documents: Iterable[str],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        """Transform documents to document-term matrix."""
        analyzer = self.build_analyzer()
        vocab = self.vocabulary_
        n_features = len(vocab)

        indptr = [0]
        indices: List[int] = []
        values: List[float] = []

        docs_list = list(raw_documents)
        n_docs = len(docs_list)

        for doc in docs_list:
            counts = Counter(analyzer(doc))
            row_items = []
            for term, count in counts.items():
                if term in vocab:
                    col = vocab[term]
                    val = 1.0 if self.binary else float(count)
                    row_items.append((col, val))
            row_items.sort(key=lambda x: x[0])

            for col, val in row_items:
                indices.append(col)
                values.append(val)
            indptr.append(len(indices))

        csr = CSRTensor(indptr, indices, values, shape=(n_docs, n_features))
        if return_sparse:
            return csr
        return csr.to_dense()

    def fit_transform(
        self,
        raw_documents: Iterable[str],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        """Learn the vocabulary dictionary and return document-term matrix."""
        docs = list(raw_documents)
        return self.fit(docs).transform(docs, return_sparse=return_sparse)

    def inverse_transform(self, X: Union[_Tensor, CSRTensor, np.ndarray]) -> List[List[str]]:
        """Return terms per document with nonzero entries in X."""
        idx_to_term = {idx: term for term, idx in self.vocabulary_.items()}
        if isinstance(X, CSRTensor):
            results = []
            for r in range(X.shape[0]):
                start, end = X.indptr[r], X.indptr[r + 1]
                row_terms = [idx_to_term[X.indices[i]] for i in range(start, end) if X.values[i] != 0]
                results.append(row_terms)
            return results

        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        results = []
        for row in arr:
            terms = [idx_to_term[col] for col in np.flatnonzero(row) if col in idx_to_term]
            results.append(terms)
        return results

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names for transformation."""
        names = [""] * len(self.vocabulary_)
        for term, idx in self.vocabulary_.items():
            names[idx] = term
        return names


class TfidfTransformer:
    """Transform a count matrix to a normalized TF or TF-IDF representation."""

    def __init__(
        self,
        norm: Optional[str] = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
    ):
        self.norm = norm.lower() if norm else None
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.idf_: Optional[np.ndarray] = None

    def fit(self, X: Union[_Tensor, CSRTensor, np.ndarray]) -> "TfidfTransformer":
        """Learn the idf vector (global term weights)."""
        if isinstance(X, CSRTensor):
            n_samples, n_features = X.shape
            # Count nonzeros per column
            df = np.bincount(X.indices, minlength=n_features).astype(np.float64)
        else:
            arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
            n_samples, n_features = arr.shape
            df = np.sum(arr > 0, axis=0, dtype=np.float64)

        if self.use_idf:
            if self.smooth_idf:
                idf = np.log((1.0 + n_samples) / (1.0 + df)) + 1.0
            else:
                df_safe = np.where(df == 0, 1.0, df)
                idf = np.log(float(n_samples) / df_safe) + 1.0
            self.idf_ = idf
        else:
            self.idf_ = np.ones(n_features, dtype=np.float64)
        return self

    def transform(
        self,
        X: Union[_Tensor, CSRTensor, np.ndarray],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        """Transform a count matrix to a tf or tf-idf representation."""
        if isinstance(X, CSRTensor):
            n_samples, n_features = X.shape
            vals = X.values.copy()
            if self.sublinear_tf:
                vals = np.where(vals > 0, 1.0 + np.log(np.maximum(vals, 1e-12)), 0.0)
            if self.use_idf and self.idf_ is not None:
                vals = vals * self.idf_[X.indices]

            csr = CSRTensor(X.indptr, X.indices, vals, shape=(n_samples, n_features))
            dense = csr.to_dense().to_numpy()
            if self.norm == "l2":
                norms = np.linalg.norm(dense, axis=1, keepdims=True)
                dense = dense / np.maximum(norms, 1e-12)
            elif self.norm == "l1":
                norms = np.sum(np.abs(dense), axis=1, keepdims=True)
                dense = dense / np.maximum(norms, 1e-12)

            if return_sparse:
                from .sparse import coo_matrix_from_dense
                return coo_matrix_from_dense(_from_numpy(dense)).to_csr()
            return _from_numpy(np.ascontiguousarray(dense, dtype=np.float32))

        arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
        arr = arr.astype(np.float64).copy()
        if self.sublinear_tf:
            arr = np.where(arr > 0, 1.0 + np.log(np.maximum(arr, 1e-12)), 0.0)
        if self.use_idf and self.idf_ is not None:
            arr = arr * self.idf_[None, :]

        if self.norm == "l2":
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
        elif self.norm == "l1":
            norms = np.sum(np.abs(arr), axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)

        t_out = _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        if return_sparse:
            from .sparse import coo_matrix_from_dense
            return coo_matrix_from_dense(t_out).to_csr()
        return t_out

    def fit_transform(
        self,
        X: Union[_Tensor, CSRTensor, np.ndarray],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        return self.fit(X).transform(X, return_sparse=return_sparse)


class TfidfVectorizer:
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to CountVectorizer followed by TfidfTransformer.
    """

    def __init__(
        self,
        lowercase: bool = True,
        stop_words: Optional[Union[str, Iterable[str], Set[str]]] = None,
        token_pattern: str = r"(?u)\b\w\w+\b",
        ngram_range: Tuple[int, int] = (1, 1),
        max_df: Union[float, int] = 1.0,
        min_df: Union[float, int] = 1,
        max_features: Optional[int] = None,
        binary: bool = False,
        norm: Optional[str] = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
    ):
        self.count_vec = CountVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            binary=binary,
        )
        self.tfidf_trans = TfidfTransformer(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
        )

    @property
    def vocabulary_(self) -> Dict[str, int]:
        return self.count_vec.vocabulary_

    @property
    def idf_(self) -> Optional[np.ndarray]:
        return self.tfidf_trans.idf_

    def fit(self, raw_documents: Iterable[str]) -> "TfidfVectorizer":
        counts = self.count_vec.fit_transform(raw_documents, return_sparse=True)
        self.tfidf_trans.fit(counts)
        return self

    def transform(
        self,
        raw_documents: Iterable[str],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        counts = self.count_vec.transform(raw_documents, return_sparse=True)
        return self.tfidf_trans.transform(counts, return_sparse=return_sparse)

    def fit_transform(
        self,
        raw_documents: Iterable[str],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        docs = list(raw_documents)
        return self.fit(docs).transform(docs, return_sparse=return_sparse)

    def inverse_transform(self, X: Union[_Tensor, CSRTensor, np.ndarray]) -> List[List[str]]:
        return self.count_vec.inverse_transform(X)

    def get_feature_names_out(self) -> List[str]:
        return self.count_vec.get_feature_names_out()


class HashingVectorizer:
    """Stateless high-speed text vectorizer using MurmurHash3 feature hashing.

    Uses bounded memory, supports out-of-vocabulary terms gracefully, and supports streaming.
    """

    def __init__(
        self,
        n_features: int = 1048576,
        lowercase: bool = True,
        stop_words: Optional[Union[str, Iterable[str], Set[str]]] = None,
        token_pattern: str = r"(?u)\b\w\w+\b",
        ngram_range: Tuple[int, int] = (1, 1),
        alternate_sign: bool = True,
        norm: Optional[str] = "l2",
        binary: bool = False,
    ):
        self.n_features = max(1, int(n_features))
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.alternate_sign = alternate_sign
        self.norm = norm.lower() if norm else None
        self.binary = binary

        self._compiled_regex = re.compile(token_pattern)

    def _resolve_stop_words(self) -> Set[str]:
        if self.stop_words is None:
            return set()
        if isinstance(self.stop_words, str):
            if self.stop_words == "english":
                return set(ENGLISH_STOP_WORDS)
            raise ValueError(f"Unknown stop words specification '{self.stop_words}'")
        return set(self.stop_words)

    def build_analyzer(self) -> Callable[[str], List[str]]:
        regex = self._compiled_regex
        stop_set = self._resolve_stop_words()
        min_n, max_n = self.ngram_range

        def _analyze(doc: str) -> List[str]:
            if self.lowercase:
                doc = doc.lower()
            tokens = regex.findall(doc)
            if stop_set:
                tokens = [t for t in tokens if t not in stop_set]
            if min_n == 1 and max_n == 1:
                return tokens
            ngrams = []
            for n in range(min_n, max_n + 1):
                for i in range(len(tokens) - n + 1):
                    ngrams.append(" ".join(tokens[i : i + n]))
            return ngrams

        return _analyze

    def fit(self, X: Any = None) -> "HashingVectorizer":
        """Stateless: no vocabulary to learn."""
        return self

    def transform(
        self,
        raw_documents: Iterable[str],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        analyzer = self.build_analyzer()
        docs = list(raw_documents)
        n_docs = len(docs)
        n_feat = self.n_features

        indptr = [0]
        indices: List[int] = []
        values: List[float] = []

        for doc in docs:
            tokens = analyzer(doc)
            h_counts: Dict[int, float] = {}
            for t in tokens:
                h = murmurhash3_32(t)
                idx = h % n_feat
                sign = -1.0 if (self.alternate_sign and (h & 0x80000000)) else 1.0
                h_counts[idx] = h_counts.get(idx, 0.0) + sign

            sorted_indices = sorted(h_counts.keys())
            for idx in sorted_indices:
                val = h_counts[idx]
                if self.binary:
                    val = 1.0 if val != 0 else 0.0
                indices.append(idx)
                values.append(val)
            indptr.append(len(indices))

        csr = CSRTensor(indptr, indices, values, shape=(n_docs, n_feat))
        if self.norm is not None:
            # Normalize rows
            dense = csr.to_dense().to_numpy()
            if self.norm == "l2":
                norms = np.linalg.norm(dense, axis=1, keepdims=True)
                dense = dense / np.maximum(norms, 1e-12)
            elif self.norm == "l1":
                norms = np.sum(np.abs(dense), axis=1, keepdims=True)
                dense = dense / np.maximum(norms, 1e-12)
            if return_sparse:
                from .sparse import coo_matrix_from_dense
                return coo_matrix_from_dense(_from_numpy(dense)).to_csr()
            return _from_numpy(np.ascontiguousarray(dense, dtype=np.float32))

        if return_sparse:
            return csr
        return csr.to_dense()

    def fit_transform(
        self,
        raw_documents: Iterable[str],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        return self.transform(raw_documents, return_sparse=return_sparse)


class BM25Vectorizer:
    """Best Matching 25 (BM25Okapi / BM25+) probabilistic ranking and vectorizer.

    Produces length-normalized term weight vectors suitable for sparse retrieval and RAG.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.0,
        lowercase: bool = True,
        stop_words: Optional[Union[str, Iterable[str], Set[str]]] = None,
        token_pattern: str = r"(?u)\b\w\w+\b",
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = None,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.max_features = max_features

        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.avgdl_: float = 0.0
        self.doc_lens_: Optional[np.ndarray] = None
        self._compiled_regex = re.compile(token_pattern)

    def _resolve_stop_words(self) -> Set[str]:
        if self.stop_words is None:
            return set()
        if isinstance(self.stop_words, str) and self.stop_words == "english":
            return set(ENGLISH_STOP_WORDS)
        return set(self.stop_words)

    def build_analyzer(self) -> Callable[[str], List[str]]:
        regex = self._compiled_regex
        stop_set = self._resolve_stop_words()
        min_n, max_n = self.ngram_range

        def _analyze(doc: str) -> List[str]:
            if self.lowercase:
                doc = doc.lower()
            tokens = regex.findall(doc)
            if stop_set:
                tokens = [t for t in tokens if t not in stop_set]
            if min_n == 1 and max_n == 1:
                return tokens
            ngrams = []
            for n in range(min_n, max_n + 1):
                for i in range(len(tokens) - n + 1):
                    ngrams.append(" ".join(tokens[i : i + n]))
            return ngrams

        return _analyze

    def fit(self, raw_documents: Iterable[str]) -> "BM25Vectorizer":
        """Fit BM25 parameters across corpus."""
        analyzer = self.build_analyzer()
        docs = list(raw_documents)
        n_docs = len(docs)

        doc_lens = []
        term_doc_counts: Counter[str] = Counter()
        term_freqs: Counter[str] = Counter()

        for doc in docs:
            terms = analyzer(doc)
            doc_lens.append(len(terms))
            term_freqs.update(terms)
            term_doc_counts.update(set(terms))

        self.doc_lens_ = np.array(doc_lens, dtype=np.float32)
        self.avgdl_ = float(np.mean(doc_lens)) if doc_lens else 1.0

        candidates = list(term_doc_counts.keys())
        if self.max_features is not None and len(candidates) > self.max_features:
            candidates.sort(key=lambda t: (-term_freqs[t], t))
            candidates = candidates[: self.max_features]
        candidates.sort()

        self.vocabulary_ = {term: idx for idx, term in enumerate(candidates)}
        n_features = len(self.vocabulary_)

        # Robertson-Spärck Jones IDF (positive variant)
        idf = np.zeros(n_features, dtype=np.float32)
        for term, idx in self.vocabulary_.items():
            df = term_doc_counts[term]
            idf[idx] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self.idf_ = idf
        return self

    def transform(
        self,
        raw_documents: Iterable[str],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        """Vectorizes documents into BM25 weight vectors."""
        analyzer = self.build_analyzer()
        vocab = self.vocabulary_
        idf = self.idf_ if self.idf_ is not None else np.ones(len(vocab), dtype=np.float32)
        avgdl = max(self.avgdl_, 1e-6)
        k1, b, delta = self.k1, self.b, self.delta

        docs = list(raw_documents)
        n_docs = len(docs)
        n_features = len(vocab)

        indptr = [0]
        indices: List[int] = []
        values: List[float] = []

        for doc in docs:
            tokens = analyzer(doc)
            dl = len(tokens)
            counts = Counter(tokens)
            row_items = []
            for term, count in counts.items():
                if term in vocab:
                    idx = vocab[term]
                    tf = float(count)
                    # BM25 term saturation formula
                    denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                    bm25_weight = idf[idx] * ((tf * (k1 + 1.0)) / max(denom, 1e-12) + delta)
                    row_items.append((idx, float(bm25_weight)))

            row_items.sort(key=lambda x: x[0])
            for idx, w in row_items:
                indices.append(idx)
                values.append(w)
            indptr.append(len(indices))

        csr = CSRTensor(indptr, indices, values, shape=(n_docs, n_features))
        if return_sparse:
            return csr
        return csr.to_dense()

    def fit_transform(
        self,
        raw_documents: Iterable[str],
        return_sparse: bool = False,
    ) -> Union[_Tensor, CSRTensor]:
        docs = list(raw_documents)
        return self.fit(docs).transform(docs, return_sparse=return_sparse)

    def score_query(
        self,
        query: str,
        corpus_vectors: Union[_Tensor, CSRTensor, np.ndarray],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Scores a query string against precomputed corpus BM25 vectors."""
        analyzer = self.build_analyzer()
        q_tokens = analyzer(query)
        q_counts = Counter(q_tokens)

        vocab = self.vocabulary_
        q_vec = np.zeros(len(vocab), dtype=np.float32)
        for t, c in q_counts.items():
            if t in vocab:
                q_vec[vocab[t]] = float(c)

        if isinstance(corpus_vectors, CSRTensor):
            dense_corpus = corpus_vectors.to_dense().to_numpy()
        elif hasattr(corpus_vectors, "to_numpy"):
            dense_corpus = corpus_vectors.to_numpy()
        else:
            dense_corpus = np.asarray(corpus_vectors)

        scores = dense_corpus.dot(q_vec)
        ranked_indices = np.argsort(-scores)
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        return [(int(i), float(scores[i])) for i in ranked_indices]


class Word2VecVectorizer:
    """Semantic document vectorizer aggregating token embeddings via mean, tfidf, or SIF pooling."""

    def __init__(
        self,
        embedding_dim: int = 64,
        pooling: str = "mean",
        lowercase: bool = True,
        sif_param: float = 1e-3,
        normalize: bool = True,
        word_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.embedding_dim = max(4, int(embedding_dim))
        self.pooling = pooling.lower()
        self.lowercase = lowercase
        self.sif_param = float(sif_param)
        self.normalize = normalize

        self.embeddings_: Dict[str, np.ndarray] = dict(word_embeddings) if word_embeddings else {}
        self.token_probs_: Dict[str, float] = {}
        self.first_principal_component_: Optional[np.ndarray] = None

    def fit(self, raw_documents: Iterable[str]) -> "Word2VecVectorizer":
        """Learn word co-occurrence representations (PPMI + SVD) if embeddings not provided."""
        docs = list(raw_documents)
        tokenized_docs = []
        token_counter = Counter()

        for doc in docs:
            text = doc.lower() if self.lowercase else doc
            tokens = re.findall(r"\w+", text)
            tokenized_docs.append(tokens)
            token_counter.update(tokens)

        total_tokens = sum(token_counter.values())
        self.token_probs_ = {t: cnt / max(1, total_tokens) for t, cnt in token_counter.items()}

        if not self.embeddings_:
            # Build vocabulary of most common words
            vocab = [t for t, _ in token_counter.most_common(2000)]
            v_to_i = {w: i for i, w in enumerate(vocab)}
            V = len(vocab)
            if V == 0:
                return self

            # Co-occurrence window 5
            co_oc = np.zeros((V, V), dtype=np.float64)
            for tokens in tokenized_docs:
                for idx, t in enumerate(tokens):
                    if t not in v_to_i:
                        continue
                    i = v_to_i[t]
                    window = tokens[max(0, idx - 5) : min(len(tokens), idx + 6)]
                    for ctx in window:
                        if ctx in v_to_i and ctx != t:
                            j = v_to_i[ctx]
                            co_oc[i, j] += 1.0

            # PPMI
            total_sum = np.sum(co_oc)
            if total_sum > 0:
                row_sum = np.sum(co_oc, axis=1, keepdims=True)
                col_sum = np.sum(co_oc, axis=0, keepdims=True)
                expected = (row_sum @ col_sum) / total_sum
                with np.errstate(divide="ignore", invalid="ignore"):
                    pmi = np.log((co_oc * total_sum) / np.maximum(expected, 1e-12))
                ppmi = np.maximum(0.0, pmi)
            else:
                ppmi = co_oc

            # Truncated SVD for latent semantic vectors
            k = min(self.embedding_dim, V)
            u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
            emb_matrix = u[:, :k] * np.sqrt(s[:k])
            if k < self.embedding_dim:
                pad = np.zeros((V, self.embedding_dim - k), dtype=np.float32)
                emb_matrix = np.hstack([emb_matrix, pad])

            for word, i in v_to_i.items():
                self.embeddings_[word] = emb_matrix[i].astype(np.float32)

        return self

    def transform(self, raw_documents: Iterable[str]) -> _Tensor:
        """Map text documents into dense semantic vectors."""
        docs = list(raw_documents)
        out_vecs = []

        for doc in docs:
            text = doc.lower() if self.lowercase else doc
            tokens = re.findall(r"\w+", text)
            vecs = [self.embeddings_[t] for t in tokens if t in self.embeddings_]

            if not vecs:
                out_vecs.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue

            vec_stack = np.array(vecs, dtype=np.float32)

            if self.pooling == "max":
                doc_vec = np.max(vec_stack, axis=0)
            elif self.pooling == "sif":
                # Arora et al. SIF: a / (a + p(w))
                weights = [self.sif_param / (self.sif_param + self.token_probs_.get(t, 1e-4)) for t in tokens if t in self.embeddings_]
                w_arr = np.array(weights, dtype=np.float32)[:, None]
                doc_vec = np.sum(vec_stack * w_arr, axis=0) / np.maximum(np.sum(w_arr), 1e-9)
            else:
                # mean pooling
                doc_vec = np.mean(vec_stack, axis=0)

            if self.normalize:
                norm = np.linalg.norm(doc_vec)
                doc_vec = doc_vec / max(norm, 1e-12)

            out_vecs.append(doc_vec)

        res = np.array(out_vecs, dtype=np.float32)
        return _from_numpy(np.ascontiguousarray(res))

    def fit_transform(self, raw_documents: Iterable[str]) -> _Tensor:
        return self.fit(raw_documents).transform(raw_documents)


class PositionalSpectralVectorizer:
    """Fourier / Harmonic spectral positional vectorizer mapping token sequences to spectral coordinates."""

    def __init__(
        self,
        num_frequencies: int = 16,
        max_seq_len: int = 512,
        basis: str = "fourier",
    ):
        self.num_frequencies = max(1, int(num_frequencies))
        self.max_seq_len = max(1, int(max_seq_len))
        self.basis = basis.lower()
        self.dim = self.num_frequencies * 2

    def encode_positions(self, positions: Union[Sequence[int], np.ndarray, _Tensor]) -> _Tensor:
        """Encodes an array of 1D sequence positions into multi-frequency spectral vectors."""
        if hasattr(positions, "to_numpy"):
            pos_arr = positions.to_numpy()
        else:
            pos_arr = np.asarray(positions, dtype=np.float32)

        L = float(self.max_seq_len)
        freqs = [2.0 ** k for k in range(self.num_frequencies)]

        specs = []
        for f in freqs:
            angles = math.pi * f * (pos_arr / L)
            specs.append(np.sin(angles))
            specs.append(np.cos(angles))

        stacked = np.stack(specs, axis=-1)
        return _from_numpy(np.ascontiguousarray(stacked, dtype=np.float32))

    def transform_sequence_lengths(self, seq_lengths: Sequence[int]) -> _Tensor:
        """Transforms a batch of sequence lengths into fixed-dimensional spectral signatures."""
        return self.encode_positions(seq_lengths)


class ContextualPoolingVectorizer:
    """Differentiable neural representation pooling for Transformer / LLM contextual states."""

    def __init__(
        self,
        pooling: str = "mean",
        normalize: bool = True,
    ):
        self.pooling = pooling.lower()
        self.normalize = normalize

    def __call__(
        self,
        hidden_states: _Tensor,
        attention_mask: Optional[Union[_Tensor, np.ndarray]] = None,
    ) -> _Tensor:
        return self.forward(hidden_states, attention_mask)

    def forward(
        self,
        hidden_states: _Tensor,
        attention_mask: Optional[Union[_Tensor, np.ndarray]] = None,
    ) -> _Tensor:
        """Pools contextual hidden states [B, S, D] into document vectors [B, D]."""
        b, s, d = hidden_states.shape

        if attention_mask is None:
            # Default to all-ones mask
            mask_t = _Tensor.ones([b, s, 1])
        else:
            mask_np = attention_mask.to_numpy() if hasattr(attention_mask, "to_numpy") else np.asarray(attention_mask)
            if mask_np.ndim == 2:
                mask_np = mask_np[:, :, None]
            mask_t = _from_numpy(np.ascontiguousarray(mask_np, dtype=np.float32))

        if self.pooling == "cls":
            pooled = hidden_states.narrow(1, 0, 1).squeeze(1)
        elif self.pooling == "max":
            # Mask out padding with large negative value
            inv_mask = (1.0 - mask_t) * -1e9
            masked_h = hidden_states * mask_t + inv_mask
            arr = masked_h.to_numpy()
            max_arr = np.max(arr, axis=1)
            pooled = _from_numpy(np.ascontiguousarray(max_arr, dtype=np.float32))
        else:
            # mean pooling
            weighted = hidden_states * mask_t
            sum_h = weighted.sum_axes([1], False)
            sum_m = mask_t.sum_axes([1], False)
            # Avoid division by zero
            sum_m_safe = sum_m + 1e-9
            pooled = sum_h / sum_m_safe

        if self.normalize:
            norm = (pooled.square().sum_axes([-1], True) + 1e-12).sqrt()
            pooled = pooled / norm

        return pooled


class SubwordTfidfVectorizer:
    """Byte-level Subword Byte-Pair Encoding (BPE) TF-IDF Vectorizer.

    Segments documents into learned or pre-trained BPE subword tokens, computes
    subword document frequencies and smooth IDF, and outputs continuous fixed-dimensional
    TF-IDF vectors. Completely avoids out-of-vocabulary (OOV) tokens.
    """

    def __init__(
        self,
        target_vocab_size: int = 500,
        tokenizer: Optional[BpeTokenizer] = None,
        sublinear_tf: bool = False,
        norm: Optional[str] = "l2",
        dtype: str = "float32",
    ) -> None:
        self.target_vocab_size = target_vocab_size
        self.tokenizer = tokenizer
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.dtype = dtype
        self.idf_: Optional[np.ndarray] = None
        self.vocab_size_: int = 0

    def fit(self, raw_documents: Sequence[str]) -> "SubwordTfidfVectorizer":
        if self.tokenizer is None:
            corpus = " ".join(raw_documents)
            self.tokenizer = BpeTokenizer.train(corpus, self.target_vocab_size)
        self.vocab_size_ = self.tokenizer.vocab_size()

        n_docs = len(raw_documents)
        df = np.zeros(self.vocab_size_, dtype=np.float64)
        for doc in raw_documents:
            ids = self.tokenizer.encode(doc)
            unique_ids = set(ids)
            for token_id in unique_ids:
                if token_id < self.vocab_size_:
                    df[token_id] += 1.0

        # Smooth IDF: ln((1 + N) / (1 + df)) + 1.0
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        return self

    def fit_transform(self, raw_documents: Sequence[str]) -> _Tensor:
        return self.fit(raw_documents).transform(raw_documents)

    def transform(self, raw_documents: Sequence[str]) -> _Tensor:
        if self.tokenizer is None or self.idf_ is None:
            raise RuntimeError("SubwordTfidfVectorizer must be fitted before transforming.")

        n_docs = len(raw_documents)
        matrix = np.zeros((n_docs, self.vocab_size_), dtype=np.float32)
        for i, doc in enumerate(raw_documents):
            ids = self.tokenizer.encode(doc)
            counts = Counter(ids)
            for token_id, count in counts.items():
                if token_id < self.vocab_size_:
                    tf = (1.0 + math.log(count)) if self.sublinear_tf else float(count)
                    matrix[i, token_id] = tf * self.idf_[token_id]

            if self.norm == "l2":
                norm = float(np.linalg.norm(matrix[i]))
                if norm > 0:
                    matrix[i] /= norm
            elif self.norm == "l1":
                norm = float(np.sum(np.abs(matrix[i])))
                if norm > 0:
                    matrix[i] /= norm

        return _from_numpy(np.ascontiguousarray(matrix, dtype=np.float32))


class Doc2VecDBOWVectorizer:
    """Distributed Bag of Words (PV-DBOW, Le & Mikolov 2014) Document Vectorizer.

    Learns continuous dense vector representations for variable-length documents
    by predicting constituent words using negative sampling and stochastic gradient descent.
    """

    def __init__(
        self,
        vector_size: int = 64,
        epochs: int = 15,
        lr: float = 0.05,
        negative_samples: int = 5,
        min_count: int = 1,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.epochs = epochs
        self.lr = lr
        self.negative_samples = negative_samples
        self.min_count = min_count
        self.seed = seed
        self.doc_vectors_: Optional[np.ndarray] = None
        self.word_vectors_: Optional[np.ndarray] = None
        self.word2id_: Dict[str, int] = {}
        self.id2word_: List[str] = []
        self.word_freq_: np.ndarray = np.array([], dtype=np.float64)

    def _tokenize(self, doc: str) -> List[str]:
        return [w.lower() for w in re.findall(r"(?u)\b\w+\b", doc)]

    def fit(self, raw_documents: Sequence[str]) -> "Doc2VecDBOWVectorizer":
        rng = np.random.RandomState(self.seed)
        tokenized_docs = [self._tokenize(d) for d in raw_documents]

        # Word vocabulary building
        word_counts: Counter[str] = Counter()
        for doc in tokenized_docs:
            word_counts.update(doc)

        self.id2word_ = [w for w, c in word_counts.items() if c >= self.min_count]
        self.word2id_ = {w: i for i, w in enumerate(self.id2word_)}
        v_size = len(self.id2word_)
        if v_size == 0:
            raise ValueError("Empty vocabulary after min_count filtering.")

        # Word sampling distribution (unigram ^ 0.75)
        raw_freqs = np.array([word_counts[w] for w in self.id2word_], dtype=np.float64)
        powered = raw_freqs ** 0.75
        self.word_freq_ = powered / powered.sum()

        n_docs = len(tokenized_docs)
        # Initialize doc and word embeddings
        self.doc_vectors_ = (rng.randn(n_docs, self.vector_size) * 0.1).astype(np.float32)
        self.word_vectors_ = (rng.randn(v_size, self.vector_size) * 0.1).astype(np.float32)

        # Training loop via SGNS
        for epoch in range(self.epochs):
            alpha = self.lr * (1.0 - epoch / float(self.epochs))
            for doc_id, tokens in enumerate(tokenized_docs):
                token_ids = [self.word2id_[t] for t in tokens if t in self.word2id_]
                if not token_ids:
                    continue
                d_vec = self.doc_vectors_[doc_id]
                for target_id in token_ids:
                    # Positive sample
                    w_vec = self.word_vectors_[target_id]
                    dot = float(np.dot(d_vec, w_vec))
                    sig = 1.0 / (1.0 + math.exp(-max(min(dot, 15.0), -15.0)))
                    grad_d = (1.0 - sig) * w_vec
                    grad_w = (1.0 - sig) * d_vec

                    # Negative samples
                    neg_ids = rng.choice(v_size, size=self.negative_samples, p=self.word_freq_)
                    for neg_id in neg_ids:
                        w_neg = self.word_vectors_[neg_id]
                        neg_dot = float(np.dot(d_vec, w_neg))
                        neg_sig = 1.0 / (1.0 + math.exp(-max(min(neg_dot, 15.0), -15.0)))
                        grad_d -= neg_sig * w_neg
                        self.word_vectors_[neg_id] -= alpha * neg_sig * d_vec

                    d_vec += alpha * grad_d
                    self.word_vectors_[target_id] += alpha * grad_w
                self.doc_vectors_[doc_id] = d_vec

        # Normalize doc vectors
        norms = np.linalg.norm(self.doc_vectors_, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors_ /= norms
        return self

    def fit_transform(self, raw_documents: Sequence[str]) -> _Tensor:
        self.fit(raw_documents)
        assert self.doc_vectors_ is not None
        return _from_numpy(np.ascontiguousarray(self.doc_vectors_, dtype=np.float32))

    def infer_vector(self, doc: str, epochs: int = 10, lr: float = 0.05) -> np.ndarray:
        """Infers a document embedding for a new, unseen document."""
        if self.word_vectors_ is None:
            raise RuntimeError("Doc2VecDBOWVectorizer must be fitted first.")
        tokens = self._tokenize(doc)
        token_ids = [self.word2id_[t] for t in tokens if t in self.word2id_]
        if not token_ids:
            return np.zeros(self.vector_size, dtype=np.float32)

        rng = np.random.RandomState(self.seed + len(tokens))
        d_vec = (rng.randn(self.vector_size) * 0.1).astype(np.float32)
        v_size = len(self.id2word_)

        for ep in range(epochs):
            alpha = lr * (1.0 - ep / float(epochs))
            for target_id in token_ids:
                w_vec = self.word_vectors_[target_id]
                dot = float(np.dot(d_vec, w_vec))
                sig = 1.0 / (1.0 + math.exp(-max(min(dot, 15.0), -15.0)))
                grad_d = (1.0 - sig) * w_vec

                neg_ids = rng.choice(v_size, size=self.negative_samples, p=self.word_freq_)
                for neg_id in neg_ids:
                    w_neg = self.word_vectors_[neg_id]
                    neg_dot = float(np.dot(d_vec, w_neg))
                    neg_sig = 1.0 / (1.0 + math.exp(-max(min(neg_dot, 15.0), -15.0)))
                    grad_d -= neg_sig * w_neg

                d_vec += alpha * grad_d

        norm = float(np.linalg.norm(d_vec))
        if norm > 0:
            d_vec /= norm
        return d_vec

    def transform(self, raw_documents: Sequence[str]) -> _Tensor:
        vectors = [self.infer_vector(doc) for doc in raw_documents]
        arr = np.vstack(vectors).astype(np.float32)
        return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


class DictVectorizer:
    """Transforms lists of feature-value mappings to vectors.

    Categorical string/bool features are converted to one-hot binary indicators,
    while numeric features (int, float) are preserved as continuous numeric values.
    Supports sparse (`CSRTensor`) and dense (`Tensor`) representations.
    """

    def __init__(self, sparse: bool = False, sort: bool = True, dtype: str = "float32") -> None:
        self.sparse = sparse
        self.sort = sort
        self.dtype = dtype
        self._native = FastDictVectorizer(sort=self.sort)
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []

    def fit(self, X: Iterable[Dict[str, Any]]) -> "DictVectorizer":
        sample_list = [dict(d) for d in X]
        self._native.fit(sample_list)
        self.feature_names_ = self._native.feature_names
        self.vocabulary_ = {name: i for i, name in enumerate(self.feature_names_)}
        return self

    def fit_transform(self, X: Sequence[Dict[str, Any]]) -> Union[_Tensor, CSRTensor]:
        return self.fit(X).transform(X)

    def transform(self, X: Sequence[Dict[str, Any]]) -> Union[_Tensor, CSRTensor]:
        sample_list = [dict(d) for d in X]
        if self.sparse:
            sparse_csr = self._native.transform_sparse(sample_list)
            return CSRTensor(
                sparse_csr.row_ptr(),
                sparse_csr.col_idx(),
                sparse_csr.values(),
                sparse_csr.shape,
            )
        else:
            return self._native.transform_dense(sample_list)

    def inverse_transform(self, X: Union[_Tensor, CSRTensor, np.ndarray]) -> List[Dict[str, Any]]:
        if hasattr(X, "to_numpy"):
            arr = X.to_numpy()
        elif hasattr(X, "to_dense"):
            arr = X.to_dense().to_numpy()
        else:
            arr = np.asarray(X)

        results = []
        for row in arr:
            d = {}
            for col_idx, val in enumerate(row):
                if val != 0:
                    fname = self.feature_names_[col_idx]
                    if "=" in fname:
                        k, v = fname.split("=", 1)
                        if v.lower() == "true":
                            d[k] = True
                        elif v.lower() == "false":
                            d[k] = False
                        else:
                            d[k] = v
                    else:
                        d[fname] = float(val)
            results.append(d)
        return results


class OneHotVectorizer:
    """Encodes categorical features as one-hot numeric arrays.

    Handles 1D sequences or 2D tables of categorical tokens.
    """

    def __init__(self, handle_unknown: str = "ignore", drop_first: bool = False) -> None:
        self.handle_unknown = handle_unknown
        self.drop_first = drop_first
        self._native = FastOneHotVectorizer(
            drop_first=self.drop_first,
            ignore_unknown=(self.handle_unknown != "error"),
        )
        self.categories_: List[List[str]] = []
        self.cat_to_idx_: List[Dict[str, int]] = []

    def fit(self, X: Sequence[Any]) -> "OneHotVectorizer":
        data = np.asarray(X)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        table = [[str(v) for v in row] for row in data]
        self._native.fit(table)
        self.categories_ = self._native.categories
        self.cat_to_idx_ = [{c: i for i, c in enumerate(cats)} for cats in self.categories_]
        return self

    def fit_transform(self, X: Sequence[Any]) -> _Tensor:
        return self.fit(X).transform(X)

    def transform(self, X: Sequence[Any]) -> _Tensor:
        data = np.asarray(X)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        table = [[str(v) for v in row] for row in data]
        return self._native.transform(table)


class DiscretizationVectorizer:
    """Discretizes continuous numerical features into discrete interval bins."""

    def __init__(self, n_bins: int = 5, strategy: str = "uniform", one_hot: bool = True) -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self.one_hot = one_hot
        self.bin_edges_: List[np.ndarray] = []

    def fit(self, X: Union[Sequence[float], np.ndarray, _Tensor]) -> "DiscretizationVectorizer":
        if hasattr(X, "to_numpy"):
            arr = X.to_numpy()
        else:
            arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_cols = arr.shape[1]
        self.bin_edges_ = []
        for col in range(n_cols):
            col_vals = arr[:, col]
            min_v, max_v = float(np.min(col_vals)), float(np.max(col_vals))
            if self.strategy == "quantile":
                quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
                edges = np.quantile(col_vals, quantiles)
            else:
                edges = np.linspace(min_v, max_v, self.n_bins + 1)[1:-1]
            self.bin_edges_.append(edges)
        return self

    def fit_transform(self, X: Union[Sequence[float], np.ndarray, _Tensor]) -> _Tensor:
        return self.fit(X).transform(X)

    def transform(self, X: Union[Sequence[float], np.ndarray, _Tensor]) -> _Tensor:
        if hasattr(X, "to_numpy"):
            arr = X.to_numpy()
        else:
            arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_samples, n_cols = arr.shape
        if self.one_hot:
            out = np.zeros((n_samples, n_cols * self.n_bins), dtype=np.float32)
            for i in range(n_samples):
                for col in range(n_cols):
                    bin_idx = int(np.digitize(arr[i, col], self.bin_edges_[col]))
                    bin_idx = min(bin_idx, self.n_bins - 1)
                    out[i, col * self.n_bins + bin_idx] = 1.0
            return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))
        else:
            out = np.zeros((n_samples, n_cols), dtype=np.float32)
            for i in range(n_samples):
                for col in range(n_cols):
                    bin_idx = int(np.digitize(arr[i, col], self.bin_edges_[col]))
                    out[i, col] = min(bin_idx, self.n_bins - 1)
            return _from_numpy(np.ascontiguousarray(out, dtype=np.float32))


class JsonTreeVectorizer:
    """Hierarchical vectorizer for arbitrary nested JSON trees, API payloads, and dict structures.

    Flattens hierarchical trees into JSON-path value occurrences, applying depth-based
    exponential decay weights beta^depth and hashing into a fixed-dimensional continuous space.
    """

    def __init__(self, n_features: int = 256, decay_rate: float = 0.8, seed: int = 42) -> None:
        self.n_features = n_features
        self.decay_rate = decay_rate
        self.seed = seed
        self._native = FastJsonTreeVectorizer(
            n_features=self.n_features,
            decay_rate=self.decay_rate,
            seed=self.seed,
        )

    def _flatten(self, obj: Any, prefix: str = "", depth: int = 0) -> List[Tuple[str, float]]:
        items = []
        weight = float(self.decay_rate ** depth)
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                items.extend(self._flatten(v, p, depth + 1))
        elif isinstance(obj, (list, tuple)):
            for i, elem in enumerate(obj):
                p = f"{prefix}[{i}]"
                items.extend(self._flatten(elem, p, depth + 1))
        elif isinstance(obj, (int, float, np.number)):
            items.append((f"{prefix}:num", float(obj) * weight))
        elif isinstance(obj, bool):
            items.append((f"{prefix}:{obj}", weight))
        elif obj is None:
            items.append((f"{prefix}:null", weight))
        else:
            items.append((f"{prefix}:{str(obj)}", weight))
        return items

    def transform_single(self, json_obj: Any) -> np.ndarray:
        pairs = self._flatten(json_obj)
        return np.array(self._native.transform_entries(pairs), dtype=np.float32)

    def transform(self, documents: Sequence[Any]) -> _Tensor:
        batch = [self._flatten(d) for d in documents]
        return self._native.transform_dense(batch)


class TimeSeriesSpectralVectorizer:
    """Transforms 1D or 2D time series / financial / sensor streams into fixed-size feature vectors.

    Extracts:
    - 5 Statistical moments: Mean, standard deviation, skewness, kurtosis, peak-to-peak amplitude.
    - FFT spectral energy bands: DC, low-frequency power, mid-frequency power, high-frequency power, spectral centroid, spectral flatness.
    - Autocorrelation coefficients at lags [1, 2, 5, 10].
    - Zero-crossing rate (ZCR).
    """

    def __init__(self, n_fft_bands: int = 4, normalize: bool = True) -> None:
        self.n_fft_bands = n_fft_bands
        self.normalize = normalize
        self._native = FastTimeSeriesSpectralVectorizer(
            n_fft_bands=self.n_fft_bands,
            normalize=self.normalize,
        )

    def transform_series(self, series: np.ndarray) -> np.ndarray:
        x = [float(v) for v in np.asarray(series, dtype=np.float64).flatten()]
        return np.array(self._native.transform_series(x), dtype=np.float32)

    def transform(self, time_series: Sequence[Any]) -> _Tensor:
        batch = [[float(v) for v in np.asarray(s, dtype=np.float64).flatten()] for s in time_series]
        return self._native.transform(batch)


class PathSignatureVectorizer:
    """Computes truncated rough path signatures (Chen's iterated integrals) for trajectories.

    Encodes non-commutative geometric ordering and path curvature (Lévy areas),
    independent of time re-parameterization.
    """

    def __init__(self, depth: int = 2) -> None:
        if depth not in (1, 2):
            raise ValueError("depth must be 1 or 2")
        self.depth = depth
        self._native = FastPathSignatureVectorizer(depth=self.depth)

    def signature_single(self, path: np.ndarray) -> np.ndarray:
        p = np.asarray(path, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        t_steps, d = p.shape
        p_list = [[float(v) for v in row] for row in p]
        return np.array(self._native.signature_single(p_list, d), dtype=np.float32)

    def transform(self, paths: Sequence[Any]) -> _Tensor:
        parsed = [np.asarray(p, dtype=np.float64) for p in paths]
        d = parsed[0].shape[1] if parsed[0].ndim > 1 else 1
        batch = [
            [[float(v) for v in (row if p.ndim > 1 else [row])] for row in p]
            for p in parsed
        ]
        return self._native.transform(batch, d)


class SphericalHarmonicsVectorizer:
    """Transforms 3D direction coordinates (x, y, z) into real spherical harmonics basis vectors Y_l^m."""

    def __init__(self, degree: int = 2) -> None:
        self.degree = degree
        self._fast_sh = FastSphericalHarmonics(degree)

    def transform(self, coordinates: Union[Sequence[Sequence[float]], np.ndarray, _Tensor]) -> _Tensor:
        if isinstance(coordinates, _Tensor):
            return self._fast_sh.transform(coordinates)
        arr = np.asarray(coordinates, dtype=np.float64)
        t = _from_numpy(np.ascontiguousarray(arr, dtype=np.float64))
        return self._fast_sh.transform(t)


class DeepSetsVectorizer:
    """Permutation-invariant vectorizer for unordered sets and multi-sets of items.

    Applies element-wise projection phi(x) followed by permutation-invariant
    aggregation rho(bigoplus_{x in X} phi(x)) with pooling in ('sum', 'mean', 'max', 'lse').
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 32,
        pooling: str = "mean",
        seed: int = 42,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.pooling = pooling
        self.seed = seed
        self._native = FastDeepSetsVectorizer(
            in_features=self.in_features,
            out_features=self.out_features,
            pooling=self.pooling,
            seed=self.seed,
        )

    def transform_set(self, elements: Union[Sequence[Sequence[float]], np.ndarray]) -> np.ndarray:
        elems = np.asarray(elements, dtype=np.float64)
        if elems.ndim == 1:
            elems = elems.reshape(1, -1)
        e_list = [[float(v) for v in row] for row in elems]
        return np.array(self._native.transform_set(e_list), dtype=np.float32)

    def transform(self, sets: Sequence[Any]) -> _Tensor:
        batch = []
        for s in sets:
            elems = np.asarray(s, dtype=np.float64)
            if elems.ndim == 1:
                elems = elems.reshape(1, -1)
            batch.append([[float(v) for v in row] for row in elems])
        return self._native.transform(batch)


class WeisfeilerLehmanVectorizer:
    """1-Weisfeiler-Lehman (1-WL) graph subtree pattern vectorizer.

    Encodes graph topology and discrete node attributes into a fixed-size histogram
    of multi-hop neighborhood subtree patterns.
    """

    def __init__(self, iterations: int = 3, n_features: int = 128, seed: int = 42) -> None:
        self.iterations = iterations
        self.n_features = n_features
        self.seed = seed
        self._native = FastWeisfeilerLehmanVectorizer(
            iterations=self.iterations,
            n_features=self.n_features,
            seed=self.seed,
        )

    def transform_graph(
        self, adj_matrix: np.ndarray, node_labels: Optional[List[str]] = None
    ) -> np.ndarray:
        adj = np.asarray(adj_matrix, dtype=np.int32)
        n = adj.shape[0]
        if node_labels is None:
            node_labels = [str(int(np.sum(adj[i]))) for i in range(n)]
        adj_list = [[int(v) for v in row] for row in adj]
        return np.array(self._native.transform_graph(adj_list, [str(l) for l in node_labels]), dtype=np.float32)

    def transform(self, graphs: Sequence[Any]) -> _Tensor:
        batch = []
        for item in graphs:
            if isinstance(item, tuple):
                adj, labels = item
            else:
                adj, labels = item, None
            adj_arr = np.asarray(adj, dtype=np.int32)
            n = adj_arr.shape[0]
            if labels is None:
                labels = [str(int(np.sum(adj_arr[i]))) for i in range(n)]
            batch.append(([[int(v) for v in row] for row in adj_arr], [str(l) for l in labels]))
        return self._native.transform_dense(batch)


class ColBERTMaxSimVectorizer:
    """ColBERT-style multi-vector late-interaction vectorizer with MaxSim scoring.

    Extracts sequences of L2-normalized token embeddings and computes token-level MaxSim matching.
    """

    def __init__(self, dim: int = 32, seed: int = 42) -> None:
        self.dim = dim
        self.seed = seed

    def encode_tokens(self, tokens: Sequence[str]) -> _Tensor:
        if not tokens:
            return _Tensor.zeros([1, self.dim])
        vecs = []
        for t in tokens:
            h1 = murmurhash3_32(t, self.seed)
            h2 = murmurhash3_32(t, self.seed + 1)
            rng = np.random.RandomState(h1 ^ h2)
            v = rng.randn(self.dim).astype(np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v /= norm
            vecs.append(v)
        arr = np.vstack(vecs).astype(np.float32)
        return _from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    def score_maxsim(self, query_tokens: _Tensor, doc_tokens: _Tensor) -> float:
        return FastMaxSimScorer.score_maxsim(query_tokens, doc_tokens)

    def batch_maxsim(self, queries: Sequence[_Tensor], docs: Sequence[_Tensor]) -> _Tensor:
        return FastMaxSimScorer.batch_maxsim(list(queries), list(docs))


class ByteHistogramVectorizer:
    """256-dimensional UTF-8 raw byte histogram vectorizer."""

    def __init__(self, log_scale: bool = True, norm: Optional[str] = "l2") -> None:
        self._fast = FastByteHistogramVectorizer(log_scale=log_scale, norm=norm)

    def transform(self, documents: Sequence[Union[str, bytes]]) -> _Tensor:
        docs_str = [
            d.decode("utf-8", errors="replace") if isinstance(d, bytes) else str(d)
            for d in documents
        ]
        return self._fast.transform_dense(docs_str)


class TopicModelVectorizer:
    """Topic Modeling vectorizer via Latent Semantic Analysis (LSA) and Non-negative Matrix Factorization (NMF).

    Projects document-term representations into K latent semantic topic components.
    """

    def __init__(
        self,
        n_components: int = 10,
        method: str = "lsa",
        max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.method = method.lower()
        self.max_iter = max_iter
        self.seed = seed
        self.tfidf = TfidfVectorizer(norm="l2", sublinear_tf=True)
        self.components_: Optional[np.ndarray] = None

    def fit(self, raw_documents: Sequence[str]) -> "TopicModelVectorizer":
        X = self.tfidf.fit_transform(raw_documents).to_numpy()
        n_samples, n_features = X.shape
        k = min(self.n_components, n_samples, n_features)
        self.n_components = k

        if self.method == "nmf":
            rng = np.random.RandomState(self.seed)
            W = np.abs(rng.randn(n_samples, k).astype(np.float64)) + 0.01
            H = np.abs(rng.randn(k, n_features).astype(np.float64)) + 0.01
            V = np.maximum(X, 0.0)

            for _ in range(self.max_iter):
                num_h = np.dot(W.T, V)
                den_h = np.dot(np.dot(W.T, W), H) + 1e-9
                H *= num_h / den_h

                num_w = np.dot(V, H.T)
                den_w = np.dot(W, np.dot(H, H.T)) + 1e-9
                W *= num_w / den_w

            self.components_ = H
        else:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            self.components_ = Vt[:k]

        return self

    def fit_transform(self, raw_documents: Sequence[str]) -> _Tensor:
        return self.fit(raw_documents).transform(raw_documents)

    def transform(self, raw_documents: Sequence[str]) -> _Tensor:
        if self.components_ is None:
            raise RuntimeError("TopicModelVectorizer must be fitted before transforming.")
        X = self.tfidf.transform(raw_documents).to_numpy()
        if self.method == "nmf":
            H = self.components_
            W = np.dot(X, np.linalg.pinv(H))
            W = np.maximum(W, 0.0)
            norm = np.sum(W, axis=1, keepdims=True) + 1e-12
            W /= norm
            return _from_numpy(np.ascontiguousarray(W, dtype=np.float32))
        else:
            Z = np.dot(X, self.components_.T)
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            mask = norms > 1e-6
            Z = np.where(mask, Z / np.maximum(norms, 1e-12), 0.0)
            return _from_numpy(np.ascontiguousarray(Z, dtype=np.float32))


__all__ = [
    "ENGLISH_STOP_WORDS",
    "murmurhash3_32",
    "fast_murmurhash3_32",
    "CountVectorizer",
    "TfidfTransformer",
    "TfidfVectorizer",
    "HashingVectorizer",
    "BM25Vectorizer",
    "Word2VecVectorizer",
    "PositionalSpectralVectorizer",
    "ContextualPoolingVectorizer",
    "SubwordTfidfVectorizer",
    "Doc2VecDBOWVectorizer",
    "DictVectorizer",
    "OneHotVectorizer",
    "DiscretizationVectorizer",
    "JsonTreeVectorizer",
    "TimeSeriesSpectralVectorizer",
    "PathSignatureVectorizer",
    "SphericalHarmonicsVectorizer",
    "DeepSetsVectorizer",
    "WeisfeilerLehmanVectorizer",
    "ColBERTMaxSimVectorizer",
    "ByteHistogramVectorizer",
    "TopicModelVectorizer",
    "FastBM25Index",
    "FastCountVectorizer",
    "FastTfidfVectorizer",
    "FastHashingVectorizer",
    "FastTextCharNGramVectorizer",
    "FastMinHashVectorizer",
    "FastSimHashVectorizer",
    "FastByteHistogramVectorizer",
    "FastMaxSimScorer",
    "FastSphericalHarmonics",
    "FastDiscretizer",
    "FastLsaVectorizer",
    "KernelSpectralVectorizer",
]
