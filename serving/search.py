"""
Content-based movie search engine using TF-IDF.

Converts natural language prompts into movie recommendations by computing
cosine similarity between the expanded prompt and a TF-IDF document built
for each movie (title + genres + release year).

Key design decisions
--------------------
* Genres are repeated twice in each document so they carry more weight
  than the title words (genre is the primary match signal).
* Prompt expansion maps shorthand terms ("90s", "scary", "animated") to
  the vocabulary that actually appears in the corpus, improving recall.
* `sublinear_tf=True` reduces the dominance of very common terms.
* We expose `rel_score` (0–1, relative to best match) so the frontend
  can display a meaningful "match quality" indicator.

Usage
-----
    engine = ContentSearchEngine()
    engine.fit(movies_df)                         # build index
    results = engine.search("funny 90s comedy")   # query
"""

import re
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Genre → hex colour mapping (used by the frontend to colour-code cards)
# ---------------------------------------------------------------------------
GENRE_COLORS: Dict[str, str] = {
    "Action":      "#e74c3c",
    "Adventure":   "#e67e22",
    "Animation":   "#3498db",
    "Children's":  "#2ecc71",
    "Comedy":      "#f1c40f",
    "Crime":       "#8e44ad",
    "Documentary": "#1abc9c",
    "Drama":       "#2980b9",
    "Fantasy":     "#9b59b6",
    "Film-Noir":   "#2c3e50",
    "Horror":      "#c0392b",
    "Musical":     "#d35400",
    "Mystery":     "#7f8c8d",
    "Romance":     "#e91e63",
    "Sci-Fi":      "#00bcd4",
    "Thriller":    "#607d8b",
    "War":         "#795548",
    "Western":     "#ff9800",
}
_DEFAULT_COLOR = "#6c757d"

# ---------------------------------------------------------------------------
# Prompt-expansion tables
# ---------------------------------------------------------------------------
# Each key is a regex pattern; value is the text inserted in its place.
# Decade expansions turn "90s" into all years in that decade so TF-IDF
# matches against the year token embedded in movie titles like "Toy Story (1995)".
_DECADE_EXPANSIONS: Dict[str, str] = {
    r"\b60s\b": "1960 1961 1962 1963 1964 1965 1966 1967 1968 1969",
    r"\b70s\b": "1970 1971 1972 1973 1974 1975 1976 1977 1978 1979",
    r"\b80s\b": "1980 1981 1982 1983 1984 1985 1986 1987 1988 1989",
    r"\b90s\b": "1990 1991 1992 1993 1994 1995 1996 1997 1998 1999",
    r"\b2000s\b": "2000 2001 2002 2003 2004 2005 2006 2007 2008 2009",
}

_SYNONYM_EXPANSIONS: Dict[str, str] = {
    r"\bscary\b":          "horror scary thriller",
    r"\bfunny\b":          "comedy humor humorous funny",
    r"\bkids?\b":          "children's family animation kids",
    r"\bchildren\b":       "children's family animation",
    r"\banimated?\b":      "animation animated cartoon",
    r"\bromantic?\b":      "romance romantic love",
    r"\bspace\b":          "sci-fi space galaxy astronaut",
    r"\bsuper\s*hero\b":   "action adventure comic",
    r"\bspooky\b":         "horror mystery thriller",
    r"\bdark\b":           "thriller drama film-noir crime",
    r"\blight\b":          "comedy romance family",
    r"\bmystery\b":        "mystery thriller crime",
    r"\bwhodunit\b":       "mystery crime thriller",
    r"\bheist\b":          "crime thriller action",
    r"\bpost.?apocalypt": "sci-fi thriller action",
    r"\btime\s*travel\b":  "sci-fi adventure fantasy",
    r"\bwestern\b":        "western action adventure",
    r"\bmusical\b":        "musical romance comedy",
    r"\bwar\b":            "war drama action",
    r"\bdocumentary\b":    "documentary",
    r"\bdoc\b":            "documentary",
}


class ContentSearchEngine:
    """
    TF-IDF content search over the movie catalogue.

    Each movie is indexed as a text document:
        "<clean_title> <genres_words> <genres_words> <year>"

    Genres are repeated to increase their TF-IDF weight relative to title
    words, since genre is typically the strongest match signal for a prompt
    like "funny animated movie".

    At query time, the engine:
      1. Expands the prompt (decade shortcuts, genre synonyms).
      2. Transforms the expanded prompt with the fitted vectoriser.
      3. Computes cosine similarity against all movie documents.
      4. Returns the top-K results with display metadata.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),       # unigrams + bigrams (e.g. "film noir")
            stop_words="english",
            max_features=15_000,
            sublinear_tf=True,        # log(1 + tf) — dampens high-frequency terms
        )
        self._item_indices: np.ndarray = np.array([], dtype=int)
        self._item_metadata: Dict[int, dict] = {}
        self._tfidf_matrix = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, movies: pd.DataFrame) -> None:
        """
        Build the TF-IDF index from a movies DataFrame.

        Parameters
        ----------
        movies : pd.DataFrame
            Must contain columns: ``item_idx`` (int), ``title`` (str),
            ``genres`` (str, pipe-separated e.g. ``"Action|Drama"``).
        """
        documents: List[str] = []
        indices: List[int] = []

        for _, row in movies.iterrows():
            item_idx = int(row["item_idx"])
            title    = str(row["title"])
            genres   = str(row["genres"])

            documents.append(self._build_document(title, genres))
            indices.append(item_idx)

            # Parse year from title parens: "Toy Story (1995)"
            year_match = re.search(r"\((\d{4})\)", title)
            year = int(year_match.group(1)) if year_match else 0

            genres_list = [g.strip() for g in genres.split("|") if g.strip()]
            primary_color = GENRE_COLORS.get(genres_list[0], _DEFAULT_COLOR) if genres_list else _DEFAULT_COLOR

            self._item_metadata[item_idx] = {
                "title":       title,
                "genres":      genres,
                "genres_list": genres_list,
                "year":        year,
                "color":       primary_color,
            }

        self._tfidf_matrix = self.vectorizer.fit_transform(documents)
        self._item_indices  = np.array(indices, dtype=int)
        self._fitted        = True

    def search(self, prompt: str, top_k: int = 12) -> List[dict]:
        """
        Find movies that best match a natural language prompt.

        Parameters
        ----------
        prompt : str
            Free-form description, e.g. ``"funny 90s comedy with romance"``.
        top_k : int
            Maximum number of results (default 12).

        Returns
        -------
        list[dict]
            Sorted by relevance (descending).  Each dict contains:
            ``rank``, ``item_idx``, ``title``, ``genres``, ``genres_list``,
            ``year``, ``color``, ``score`` (raw TF-IDF cosine),
            ``rel_score`` (0–1, relative to best match).
        """
        if not self._fitted:
            raise RuntimeError(
                "ContentSearchEngine has not been fitted. "
                "Call fit(movies_df) before search()."
            )

        expanded   = self._expand_prompt(prompt)
        query_vec  = self.vectorizer.transform([expanded])
        sims       = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_idx    = np.argsort(sims)[::-1][:top_k]
        max_score  = float(sims[top_idx[0]]) if len(top_idx) else 1.0

        results: List[dict] = []
        for rank, idx in enumerate(top_idx, start=1):
            item_idx  = int(self._item_indices[idx])
            raw_score = float(sims[idx])
            meta      = self._item_metadata[item_idx]

            results.append({
                "rank":        rank,
                "item_idx":    item_idx,
                "title":       meta["title"],
                "genres":      meta["genres"],
                "genres_list": meta["genres_list"],
                "year":        meta["year"],
                "color":       meta["color"],
                "score":       round(raw_score, 6),
                # Relative score: 1.0 = best result, ~0.x = weaker matches
                "rel_score":   round(raw_score / (max_score + 1e-8), 4),
            })

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_document(self, title: str, genres: str) -> str:
        """
        Build a TF-IDF document string from a movie's title and genres.

        Genres are appended twice so their terms receive higher TF weight,
        making genre-keyword queries ("comedy", "action") more reliable.
        """
        year_match  = re.search(r"\((\d{4})\)", title)
        year_str    = year_match.group(1) if year_match else ""
        clean_title = re.sub(r"\s*\(\d{4}\)\s*", " ", title).strip()
        genre_words = genres.replace("|", " ").replace("-", " ").lower()
        # Format: "title genres genres year" — genres doubled for weight
        return f"{clean_title} {genre_words} {genre_words} {year_str}".strip()

    @staticmethod
    def _expand_prompt(prompt: str) -> str:
        """
        Expand shorthand phrases to vocabulary present in the movie corpus.

        Examples
        --------
        ``"90s comedy"``   → ``"1990 1991 … 1999 comedy humor humorous"``
        ``"scary kids movie"`` → ``"horror scary thriller children's family animation"``
        """
        expanded = prompt
        all_expansions = {**_DECADE_EXPANSIONS, **_SYNONYM_EXPANSIONS}
        for pattern, replacement in all_expansions.items():
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        return expanded
