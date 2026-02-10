"""Caption-based concept identification."""

from dataclasses import dataclass
from itertools import chain
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Sequence, Iterable
from fractions import Fraction

import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

Score = float | Fraction

@dataclass(frozen=True, order=True)
class Concept:
    """Class representing a textual concept."""
    lemma: str
    pos: str

    @staticmethod
    def save_vocab_to_csv(vocabulary: Sequence['Concept'], path: str | Path):
        """Save the vocabulary to a CSV file."""
        path = Path(path)
        assert path.is_file() or not path.exists(), "Path must be a file"
        path.parent.mkdir(parents=True, exist_ok=True)
        lemmas = [concept.lemma for concept in vocabulary]
        pos = [concept.pos for concept in vocabulary]

        data_dict = {
            'pos': list(pos),
            'lemma': lemmas,
        }
        df = pd.DataFrame(data_dict)
        df.to_csv(path, index=False)
        logger.info("Saved vocabulary to %s", path)
        return path

    @staticmethod
    def load_vocab_from_csv(path: str | Path) -> Sequence['Concept']:
        """Load the extractor from disk."""
        path = Path(path)
        assert path.is_file(), "Path must be a file"
        df = pd.read_csv(path)

        parts_of_speeches = df['pos'].fillna('').tolist()
        lemmas = df['lemma'].astype(str).replace('nan', None).tolist()

        vocabulary = [
            Concept(pos=pos, lemma=lemma)
            for lemma, pos in zip(lemmas, parts_of_speeches)
        ]
        return vocabulary

class ImageConceptEncoder:
    """Converts image captions to concepts."""

    def __init__(
        self, method='average', min_threshold=1, max_threshold=1.0
    ):
        self.method = method
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.counter = Counter()
        self.lemma = nltk.stem.WordNetLemmatizer()

    @staticmethod
    def from_vocabulary(vocabulary: Iterable[Concept], **kwargs):
        """Create an encoder from a vocabulary."""
        counter = Counter(vocabulary)
        instance = ImageConceptEncoder(**kwargs)
        instance.counter = counter
        return instance

    def fit(self, data: Iterable[Iterable[str]]):
        """Fit the encoder to the data."""
        num_images = 0
        for captions in data:
            tags = self.extract_concept_sents(captions)
            self.counter.update(set(chain.from_iterable(tags)))
            num_images += 1

        # filter infrequent and overly frequent tags in the image dataset
        logger.info("Before frequency filtering, vocabulary size is: %d", len(self.counter))
        self.counter = Counter(
            {tag: count for tag, count in self.counter.items()
             if count >= self.min_threshold
             and count / num_images <= self.max_threshold}
        )
        logger.info("After frequency filtering, vocabulary size is: %d", len(self.counter))
        return self

    def transform(self, data: Iterable[Iterable[str]]) -> list[Sequence[tuple[Concept, Score]]]:
        """Transform the data to concepts."""
        return [self._captions_to_tags(captions) for captions in data]

    def _filter_vocab(self, tags: Iterable[Concept]) -> list[Concept]:
        """Filter the tags based on the fitted vocabulary."""
        assert self.counter, "Vocabulary is empty. Please call fit() first."

        # filter out tags not in the (fitted) vocabulary
        return [tag for tag in tags if tag in self.counter]

    def _captions_to_tags(self, captions: Iterable[str]) -> Sequence[tuple[Concept, Score]]:
        """Convert captions for a single image to tags."""
        assert isinstance(next(iter(captions)), str), "Captions must be strings"

        tags_all_captions = [
            self._filter_vocab(tags)
            for tags in self.extract_concept_sents(captions)
        ]

        if self.method == 'union':
            tags = list(
                set(tag for tags_single_caption in tags_all_captions for tag in tags_single_caption)
            )
            tags = [(tag, 1.0) for tag in tags]
        elif self.method == 'average':
            scores: dict[Concept, int] = defaultdict(int)
            for tags_single_caption in tags_all_captions:
                for tag in set(tags_single_caption):
                    scores[tag] += 1

            assert not scores or max(scores.values()) <= len(tags_all_captions)
            tags = [(tag, Fraction(score, len(tags_all_captions))) for tag, score in scores.items()]
        else:
            raise ValueError(f"Unknown method: {self.method}. Supported methods are: union, average.")
        return tags

    def get_vocab(self) -> list[Concept]:
        """Get the vocabulary."""
        return list(self.counter.keys())

    def _wordnet_filter(self, word: str, pos: Optional[str] = None) -> bool:
        """Returns True if the word is in WordNet."""
        synsets = wn.synsets(word, pos=pos)
        return bool(synsets)

    def extract_concept_sents(self, texts: Iterable[str]) -> list[list[Concept]]:
        """Extract concepts from sentences by POS tagging."""

        pos_tags = nltk.pos_tag_sents(nltk.word_tokenize(sent, preserve_line=True) for sent in texts)
        concepts = []
        for text, tags in zip(texts, pos_tags):
            assert isinstance(text, str), "Input must be a string"

            # lemmatize
            nouns = [self.lemma.lemmatize(word, pos=wn.NOUN) for (word, pos) in tags if pos[0] == 'N']
            adjs = [self.lemma.lemmatize(word, pos=wn.ADJ) for (word, pos) in tags if pos[0] == 'J']
            verbs = [self.lemma.lemmatize(word, pos=wn.VERB) for (word, pos) in tags if pos[0] == 'V']

            # filter out words not in WordNet
            nouns = [word for word in nouns if self._wordnet_filter(word, pos=wn.NOUN)]
            adjs = [word for word in adjs if self._wordnet_filter(word, pos=wn.ADJ)]
            verbs = [word for word in verbs if self._wordnet_filter(word, pos=wn.VERB)]

            # create concepts
            concepts.append([
                Concept(lemma=word, pos=pos)
                for word, pos in zip(
                    nouns + adjs + verbs,
                    ['n'] * len(nouns) + ['a'] * len(adjs) + ['v'] * len(verbs)
                )
            ])

        return concepts

class ConceptTensorBuilder:
    """Convert concepts to COO tensors."""
    def __init__(self, vocabulary: Sequence[Concept]):
        assert len(set(vocabulary)) == len(vocabulary), "Vocabulary must be unique"
        self.concept_to_idx = {concept: i for i, concept in enumerate(vocabulary)}

    def transform(self, data: Sequence[Sequence[tuple[Concept, Score]]]):
        """Transform the data to COO tensors."""
        size = (len(data), len(self.concept_to_idx))
        indices = ([], [])
        values = []
        for i, tags in enumerate(data):
            for tag, score in tags:
                indices[0].append(i)
                indices[1].append(self.concept_to_idx[tag])
                values.append(score)
        return indices, values, size
