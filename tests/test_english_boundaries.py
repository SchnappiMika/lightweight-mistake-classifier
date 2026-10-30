"""English inflections, negation and reading boundaries without online corpora."""
from __future__ import annotations

import pytest

from app import config
from app.core.models import FeatureBundle, Question
from app.core.preprocess import cleaner, text_en


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(text_en, "_nltk_state", False)
    monkeypatch.setattr(text_en, "_lemmatizer", None)


@pytest.mark.parametrize("word, expected", [
    ("family", "family"), ("early", "early"), ("friendly", "friendly"),
    ("quickly", "quickly"), ("only", "only"), ("during", "during"),
    ("ceiling", "ceiling"), ("building", "building"), ("hundred", "hundred"),
    ("news", "news"), ("analysis", "analysis"), ("physics", "physics"),
    ("business", "business"), ("species", "species"), ("status", "status"),
    ("always", "always"), ("perhaps", "perhaps"), ("clothes", "clothes"),
    ("heroes", "hero"), ("quizzes", "quiz"),
    ("driving", "drive"), ("received", "receive"), ("developed", "develop"),
    ("studies", "study"), ("studied", "study"), ("running", "run"),
    ("stopped", "stop"), ("planned", "plan"), ("trying", "try"),
    ("lying", "lie"), ("dying", "die"), ("scientists", "scientist"),
    ("families", "family"), ("movies", "movie"), ("boxes", "box"),
    ("children", "child"), ("written", "write"), ("bought", "buy"),
    ("driven", "drive"), ("analyses", "analysis"), ("criteria", "criterion"),
])
def test_conservative_offline_word_forms(word, expected):
    assert text_en._rule_lemmatize(word) == expected


@pytest.mark.parametrize("sentence, negations", [
    ("She can't drive and has never received a ticket.", ["not", "never"]),
    ("He cannot agree; they won't accept it.", ["not", "not"]),
    ("They didn’t leave. She isn’t early.", ["not", "not"]),
    ("No buses, nor trains, without a driver.", ["no", "nor", "without"]),
    ("Neither family has ever agreed.", ["neither"]),
])
def test_negation_survives_contractions_and_stopwords(sentence, negations):
    tokens = text_en.tokenize_en(sentence)
    assert [token for token in tokens if token in text_en._NEGATION] == negations
    assert not {"can't", "cannot", "won't", "didn", "isn"}.intersection(tokens)


def test_possessives_and_valid_words_are_preserved():
    assert text_en.tokenize_en("The family's friendly driver received early warnings.") == [
        "family", "friendly", "driver", "receive", "early", "warning"]


def test_unknown_inflections_are_not_arbitrarily_chopped():
    assert text_en.tokenize_en("unmappedly unlisted frobnicated") == [
        "unmappedly", "unlisted", "frobnicated"]


@pytest.mark.parametrize("source, kept, removed", [
    ("Which route? A. invented B. found", ["Which route?"], ["invented", "found"]),
    ("1. First question?\nA. red\nB. blue\nC. green\nD. yellow\n"
     "The family moved early.\n2. Why did they move?\nA. work\nB. study",
     ["First question?", "The family moved early.", "Why did they move?"],
     ["red", "blue", "green", "yellow", "work", "study"]),
    ("A. Smith arrived early.\nHis family received a letter.",
     ["A. Smith", "His family received a letter."], []),
    ("Passage A. The family drives.\nPassage B. The children walk.",
     ["The family drives.", "The children walk."], []),
    ("1. Why? A. work B. study 2. What did the family receive? A. letter B. parcel",
     ["Why?", "What did the family receive?"], ["work", "study", "letter", "parcel"]),
    ("1.Why? A. school B. home 2.What is the price? A. 3.14 dollars B. 8.50 dollars",
     ["Why?", "What is the price?"], ["school", "home", "dollars"]),
    ("Why?\nＡ． invent\nＢ． discover\nThe next passage concerns physics.",
     ["Why?", "The next passage concerns physics."], ["invent", "discover"]),
])
def test_choice_groups_keep_following_question_and_passage(source, kept, removed):
    result = cleaner.strip_option_bodies(source)
    for text in kept:
        assert text in result
    for text in removed:
        assert text not in result


def test_reading_feature_bundle_retains_later_passage():
    question = Question(subject=config.SUBJECT_ENGLISH, qtype=config.QTYPE_READING,
                        raw_text="Why? A. noise B. traffic\n"
                                 "The family was driving early and never received a letter.\n"
                                 "2. What happened? A. delay B. success")
    bundle = FeatureBundle()
    text_en.EnglishTextUnit().process(question, bundle)
    assert {"family", "drive", "early", "never", "receive", "letter"} <= set(bundle.text_tokens)
    assert not {"noise", "traffic", "delay", "success"}.intersection(bundle.text_tokens)
    assert "family" in bundle.clean_text
