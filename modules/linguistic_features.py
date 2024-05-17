from spellchecker import SpellChecker
import os

import pandas as pd
import numpy as np
from string import punctuation, printable
import re

import spacy
nlp = spacy.load("en_core_web_sm")

spell = SpellChecker()
# corpus = set(nlp.vocab.strings)

FEATURES = set()


def preprocess_text(text: str):
    text = text.lower()
    # text = removeHTML(text)
    text = re.sub("http\w+", '', text)  # remove urls
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
#     x = expandContractions(x)
    text = re.sub(r"\.+", ".", text)  # remove extra periods
    text = re.sub(r"\,+", ",", text)  # remove extra commas
    text = text.strip()  # remove leading and trailing spaces
    return text


def is_misspelled(words: list):
    return len([spell.unknown(word) for word in words])


def get_paragraphs(data_df: pd.DataFrame):
    data_df['paragraph'] = data_df['full_text'].apply(
        lambda x: x.split("\n\n"))

    # preprocess paragraphs
    data_df['paragraph'] = data_df['paragraph'].apply(
        lambda x: [preprocess_text(para) for para in x])

    # drop empty paragraphs
    data_df['paragraph'] = data_df['paragraph'].apply(
        lambda x: [para for para in x if para.strip()])

    return data_df


def get_sentences(data_df: pd.DataFrame):
    # nlp.add_pipe('sentencizer')
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')
    data_df['sentence'] = data_df['paragraph'].apply(
        lambda x: [i.sent for i in nlp(x).sents])
    return data_df


def get_tokens(data_df: pd.DataFrame):
    data_df['words'] = data_df['sentence'].apply(
        lambda x: [word.text for word in x if word.text])
    data_df['lemmas'] = data_df['sentence'].apply(
        lambda x: [word.lemma_ for word in x if word.text])
    data_df['pos'] = data_df['sentence'].apply(
        lambda x: [word.pos_ for word in x if word.text])
    data_df['is_stop'] = data_df['sentence'].apply(
        lambda x: [word.is_stop for word in x if word.text])

    return data_df


def get_features_in_essays(data_df: pd.DataFrame, column_name: str, feature_name: str):
    group = data_df.copy()
    new_columns = {}
    new_columns['mean_' + feature_name +
                '_in_essay'] = group[column_name].mean()
    FEATURES.add('mean_' + feature_name + '_in_essay')

    new_columns['max_' + feature_name +
                '_in_essay'] = group[column_name].max()
    FEATURES.add('max_' + feature_name + '_in_essay')

    new_columns['min_' + feature_name +
                '_in_essay'] = group[column_name].min()
    FEATURES.add('min_' + feature_name + '_in_essay')

    new_columns['25th_percentile_' + feature_name +
                '_in_essay'] = np.percentile(group[column_name], 25)
    FEATURES.add('25th_percentile_' + feature_name + '_in_essay')

    new_columns['75th_percentile_' + feature_name +
                '_in_essay'] = np.percentile(group[column_name], 75)
    FEATURES.add('75th_percentile_' + feature_name + '_in_essay')

    data_df = pd.add([data_df, pd.DataFrame(new_columns)], axis=1)

    return data_df


def get_features_in_paragraphs(data_df: pd.DataFrame, column_name: str, feature_name: str):
    new_columns = {}
    group = data_df.copy().groupby(['essay_id'])[column_name]

    new_columns['mean_' + feature_name +
                '_in_paragraph'] = group.transform('mean')
    FEATURES.add('mean_' + feature_name + '_in_paragraph')

    new_columns['max_' + feature_name +
                '_in_paragraph'] = group.transform('max')
    FEATURES.add('max_' + feature_name + '_in_paragraph')

    new_columns['min_' + feature_name +
                '_in_paragraph'] = group.transform('min')
    FEATURES.add('min_' + feature_name + '_in_paragraph')

    new_columns['25th_percentile_' + feature_name +
                '_in_paragraph'] = group.transform(lambda x: np.percentile(x, 25))
    FEATURES.add('25th_percentile_' + feature_name + '_in_paragraph')

    new_columns['75th_percentile_' + feature_name +
                '_in_paragraph'] = group.transform(lambda x: np.percentile(x, 75))
    FEATURES.add('75th_percentile_' + feature_name + '_in_paragraph')

    data_df = pd.concat([data_df, pd.DataFrame(new_columns)], axis=1)

    return data_df


def get_features_in_sentences(data_df: pd.DataFrame, column_name: str, feature_name: str):
    new_columns = {}
    group = data_df.copy().groupby(['essay_id'])[column_name]

    new_columns['mean_' + feature_name +
                '_in_sentence'] = group.transform('mean')
    FEATURES.add('mean_' + feature_name + '_in_sentence')

    new_columns['max_' + feature_name +
                '_in_sentence'] = group.transform('max')
    FEATURES.add('max_' + feature_name + '_in_sentence')

    new_columns['min_' + feature_name +
                '_in_sentence'] = group.transform('min')
    FEATURES.add('min_' + feature_name + '_in_sentence')

    new_columns['25th_percentile_' + feature_name +
                '_in_sentence'] = group.transform(lambda x: np.percentile(x, 25))
    FEATURES.add('25th_percentile_' + feature_name + '_in_sentence')

    new_columns['75th_percentile_' + feature_name +
                '_in_sentence'] = group.transform(lambda x: np.percentile(x, 75))
    FEATURES.add('75th_percentile_' + feature_name + '_in_sentence')

    data_df = pd.concat([data_df, pd.DataFrame(new_columns)], axis=1)

    return data_df


def get_features_multi_levels(data_df: pd.DataFrame, column_name: str, feature_name: str):
    data_df = get_features_in_sentences(data_df, column_name, feature_name)
    data_df[feature_name + '_in_paragraph'] = data_df.groupby(
        ['essay_id', 'paragraph'])[column_name].transform('sum')
    data_df = get_features_in_paragraphs(
        data_df, feature_name + '_in_paragraph', feature_name)
    data_df[feature_name +
            '_in_essay'] = data_df.groupby('essay_id')[column_name].transform('sum')
    FEATURES.add(feature_name + '_in_essay')

    return data_df


def get_features(data_df: pd.DataFrame,  save: bool = False, path: str = None):
    data_df = get_paragraphs(data_df).explode('paragraph')

    data_df['full_text'] = data_df['full_text'].apply(preprocess_text)

    data_df = get_sentences(data_df).explode('sentence')

    data_df = get_tokens(data_df)
    data_df['sentence'] = data_df['sentence'].apply(lambda x: x.text)

    # get paragraph features
    data_df['num_paragraphs'] = data_df.groupby(
        'essay_id')['paragraph'].transform('nunique')
    FEATURES.add('num_paragraphs')

    # get number of sentences features
    data_df['num_sents_in_paragraph'] = data_df.groupby(['essay_id', 'paragraph'])[
        'sentence'].transform('nunique')
    data_df = get_features_in_paragraphs(
        data_df, 'num_sents_in_paragraph', 'num_sentences')
    
    data_df['num_sents_in_essay'] = data_df.groupby('essay_id')[
        'sentence'].transform('nunique')

    # get number of words features
    data_df['num_words_in_sentence'] = data_df['words'].apply(len)
    data_df = get_features_multi_levels(
        data_df, 'num_words_in_sentence', 'num_words')

    # get length of words features
    data_df['mean_word_lens_in_sentence'] = data_df['words'].apply(
        lambda x: np.mean([len(word) for word in x]))
    data_df = get_features_multi_levels(
        data_df, 'mean_word_lens_in_sentence', 'mean_word_lens')

    # get number of proper nouns features
    data_df['num_proper_nouns_in_sentence'] = data_df['pos'].apply(
        lambda x: np.count_nonzero(['PROPN' in pos for pos in x]))
    data_df = get_features_multi_levels(
        data_df, 'num_proper_nouns_in_sentence', 'num_proper_nouns')

    # get number of nouns features
    data_df['num_nouns_in_sentence'] = data_df['pos'].apply(
        lambda x: np.count_nonzero(['NOUN' in pos for pos in x]))
    data_df = get_features_multi_levels(
        data_df, 'num_nouns_in_sentence', 'num_nouns')

    # get number of verbs features
    data_df['num_verbs_in_sentence'] = data_df['pos'].apply(
        lambda x: np.count_nonzero(['VERB' in pos for pos in x]))
    data_df = get_features_multi_levels(
        data_df, 'num_verbs_in_sentence', 'num_verbs')

    # get number of adjectives features
    data_df['num_adjectives_in_sentence'] = data_df['pos'].apply(
        lambda x: np.count_nonzero(['ADJ' in pos for pos in x]))
    data_df = get_features_multi_levels(
        data_df, 'num_adjectives_in_sentence', 'num_adjectives')

    # get number of adverbs features
    data_df['num_adverbs_in_sentence'] = data_df['pos'].apply(
        lambda x: np.count_nonzero(['ADV' in pos for pos in x]))
    data_df = get_features_multi_levels(
        data_df, 'num_adverbs_in_sentence', 'num_adverbs')

    # get number of pronouns features
    data_df['num_pronouns_in_sentence'] = data_df['pos'].apply(
        lambda x: np.count_nonzero(['PRON' in pos for pos in x]))
    data_df = get_features_multi_levels(
        data_df, 'num_pronouns_in_sentence', 'num_pronouns')

    # get number of conjunctions features
    data_df['num_conjunctions_in_sentence'] = data_df['pos'].apply(
        lambda x: np.count_nonzero(['CONJ' in pos for pos in x]))
    data_df = get_features_multi_levels(
        data_df, 'num_conjunctions_in_sentence', 'num_conjunctions')

    # get number of misspelled words features
    data_df['num_misspelled_words_in_sentence'] = data_df['lemmas'].apply(
        lambda x: is_misspelled(x))
    data_df = get_features_multi_levels(
        data_df, 'num_misspelled_words_in_sentence', 'num_misspelled_words')
    
    # get number of unique words features
    data_df['num_unique_words_in_sentence'] = data_df['lemmas'].apply(
        lambda x: len(set(x)))
    data_df = get_features_multi_levels(
        data_df, 'num_unique_words_in_sentence', 'num_unique_words')
    
    # get number of stop words features
    data_df['num_stop_words_in_sentence'] = data_df['is_stop'].apply(
        lambda x: np.count_nonzero(x))
    data_df = get_features_multi_levels(
        data_df, 'num_stop_words_in_sentence', 'num_stop_words')
    
    data_df = data_df[['essay_id', 'full_text',
                       'score', 'sentence'] + sorted(list(FEATURES))]

    data_df = data_df.drop_duplicates()

#     print(data_df.shape, data_df[['essay_id', 'full_text',  'score'] + list(FEATURES)].drop_duplicates().shape)

    if save:
        data_df.to_csv(path, index=False)
        with open(os.path.join(os.path.dirname(path), 'features.txt'), 'w') as f:
            for item in sorted(FEATURES):
                f.write("%s\n" % item)

    return data_df, FEATURES
