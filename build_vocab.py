# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 16:58
# @Author  : Tao
# @Project : ImageCaption
# @File    : build_vocab.py
"""
Use for build and load Vocabulary
"""

# ----- import package -----#
import argparse
import json
import nltk
import pickle
from collections import Counter
from utils.util import Vocabulary


def anno_loader(json_file):
    """
    use to load coco caption annotations
    :param json_file: annotation file abspath
    :return: anns, a dict,
        the key is the caption id and the value is annotation,
        each annotation also a dict with three keys: 'caption', 'id', 'image_id'
    """
    dataset = json.load(open(json_file, 'r'))
    anns = {}
    if 'annotations' in dataset:
        for ann in dataset['annotations']:
            anns[ann['id']] = ann
    return anns


def build_vocab(args):
    """
    build Vocabulary
    :param args: Preset parameter dic
    :return: None, it will create a dictionary
    """
    counter = Counter()
    anns = anno_loader(args.caption_path)
    ids = anns.keys()

    for i, id_ in enumerate(ids):
        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

        caption = str(anns[id_]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold',
    # then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= args.threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for _, word in enumerate(words):
        vocab.add_word(word)

    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Done! Vocabulary has been established!")


def load_pkl(file):
    """
    Note: return Vocabulary type, need to import Vocabulary class
    :param file: .pkl file path
    :return: Vocabulary
    """
    Vocabulary()
    with open(file, 'rb') as f:
        vocab = pickle.load(f)

    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='./data/captions_train2014.json',
                        help='the file of annotation')
    parser.add_argument('--threshold', type=int,
                        default=4,
                        help='Word frequency threshold')
    parser.add_argument('--vocab_path', type=str,
                        default='./vocal.pkl',
                        help='Vocabulary save file')

    args = parser.parse_args()

    build_vocab(args)

    # check pkl file(Vocabulary)
    vocab = load_pkl(args.vocab_path)
    print(vocab.word2idx['people'])
    print((vocab.idx2word[52]))
    print("Total vocabulary size: {}".format(len(vocab)))
