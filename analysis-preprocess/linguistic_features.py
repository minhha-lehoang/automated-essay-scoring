from spellchecker import SpellChecker
import os

import pandas as pd
import numpy as np
from string import punctuation, printable
import re

import spacy
nlp = spacy.load("en_core_web_sm")

spell = SpellChecker()

def count_misspelled(words):
    words = [word for word in words if word.strip()]

    words = set(words) - {"'s", "n't"} - set(punctuation)

    return spell.unknown(words)


def get_sentences(text: str):
    text = "".join(filter(lambda x: x in printable, text)).strip()
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def preprocess_text(text: str):
    text = re.sub(r'\s+', ' ', text)
    text = "".join(filter(lambda x: x in printable, text)).strip()
    return text


def get_features(data_df: pd.DataFrame, output_dir='./output/data_features.csv', save=True):
    # parse the text into paragraphs
    data_df['paragraphs'] = data_df['full_text'].apply(lambda x: x.split("\n\n"))
    data_df['num_paragraphs'] = data_df['paragraphs'].apply(len)

    data_df['full_text'] = data_df['full_text'].apply(preprocess_text)

    # into sentences
    data_df['sentences'] = data_df['full_text'].apply(get_sentences)
    data_df['num_sentences'] = data_df['sentences'].apply(len)

    # Create lists to hold the results
    words = []
    lemmas = []
    pos = []
    is_stop_word = []

    # Process the texts in batches
    for i, doc in enumerate(nlp.pipe(data_df['full_text'], batch_size=50)):
        print(f"Processing batch {i+1}/{len(data_df)}", end="\r")

        words.append([token.text for token in doc])
        lemmas.append([token.lemma_ for token in doc])
        pos.append([token.pos_ for token in doc])
        is_stop_word.append([token.is_stop for token in doc])

    data_df['words'] = words
    data_df['lemma'] = lemmas
    data_df['pos'] = pos
    data_df['is_stop_word'] = is_stop_word

    data_df['num_words'] = data_df['words'].apply(len)

    data_df['num_conjunctions'] = data_df['pos'].apply(
        lambda x: len([pos for pos in x if pos == 'CCONJ']))

    data_df['num_distinct_words'] = data_df['lemma'].apply(
        lambda x: len(set(x)))

    data_df['num_misspell'] = data_df['lemma'].apply(count_misspelled)
    data_df['num_misspell'] = data_df['num_misspell'].apply(len)

    data_df['mean_word_len'] = data_df['lemma'].apply(lambda x: np.mean(
        [len(word) for word in x if word.strip() and word not in punctuation]))

    data_df['mean_sent_len'] = data_df['sentences'].apply(
        lambda x: np.mean([len([token.text for token in nlp(sent)]) for sent in x]))

    if save:
        data_df.to_csv(output_dir, index=False)

    return data_df
