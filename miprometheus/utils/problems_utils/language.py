#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Sean Robertson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""language.py: Class that handles embedding of language problems
                Pretrained embedding handler based on vocab.py in torchtext

__author__ = "Vincent Marois, Ryan L. McAvoy"

"""

import torch
from collections import Counter, OrderedDict
from torchtext import vocab


class Language(object):
    """
    Class that loads pretrained embeddings from Torchtext.
    """

    def __init__(self, name):
        """
        Constructor.

        :param name: string to name the language (at the moment it doesn't do anything)

        """
        self.name = name
        # Choose the kind of Vocab class to call. At the moment, we are just
        # using whole word vocab as opposed to sub word tokens
        self.vocab_cls = vocab.Vocab
        self.init_token = None
        self.eos_token = None
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"

    def embed_sentence(self, sentence):
        """
        Embed an entire sentence using a pretrained embedding.

        :param sentence: A string containing the words to embed
        :returns: FloatTensor of embedded vectors [max_sentence_length, embedding size]

        """
        outsentence = torch.zeros(
            (len(sentence.split()), self.vocab.vectors.size()[1]))

        # embed a word at a time
        for i, word in enumerate(sentence.split()):
            outsentence[i, :] = self.embed_word(word)

        return outsentence

    def embed_word(self, word):
        """
        Embed a single word.

        :param sentence: A string containing a single word to embed
        :returns: FloatTensor with an single embedded vector in it [embedding size]

        """
        # convert the word to an integer index and return the corresponding
        # embedding vector
        index = self.vocab.stoi[word]
        return self.vocab.vectors[index]

    def return_index_from_word(self, word):
        """
        returns the index of a word in the vocab.

        :param word: String of word in dictionary

        """
        return self.vocab.stoi[word]

    def return_word_from_index(self, index):
        """
        Returns a word in the vocab from its index.

        :param index: integer index of the word in the dictionary

        """

        return self.vocab.itos[index]

    def build_pretrained_vocab(self, data_set, **kwargs):
        """
        Construct the torchtext Vocab object from a list of sentences. This
        allows us to load only vectors we actually need.

        :param data_set: A list containing strings (either sentences or just single word string work)
        :param \**kwargs: The keyword arguments for the vectors class from torch text. The most important kwarg is vectors which is a string containing the embedding type to be loaded

        """

        counter = Counter()

        # Break list of sentences into sentences and then count the number of
        # times a word appears
        for data in data_set:
            counter.update(data.split())

        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)


"""
The names of the classes available in torchtext vocab for reference
    "charngram.100d": partial(CharNGram),
    "fasttext.en.300d": partial(FastText, language="en"),
    "fasttext.simple.300d": partial(FastText, language="simple"),
    "glove.42B.300d": partial(GloVe, name="42B", dim="300"),
    "glove.840B.300d": partial(GloVe, name="840B", dim="300"),
    "glove.twitter.27B.25d": partial(GloVe, name="twitter.27B", dim="25"),
    "glove.twitter.27B.50d": partial(GloVe, name="twitter.27B", dim="50"),
    "glove.twitter.27B.100d": partial(GloVe, name="twitter.27B", dim="100"),
    "glove.twitter.27B.200d": partial(GloVe, name="twitter.27B", dim="200"),
    "glove.6B.50d": partial(GloVe, name="6B", dim="50"),
    "glove.6B.100d": partial(GloVe, name="6B", dim="100"),
    "glove.6B.200d": partial(GloVe, name="6B", dim="200"),
    "glove.6B.300d": partial(GloVe, name="6B", dim="300")
"""

# unit test of vocab
# pretrained embedding
# Based on the example of
# https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb

if __name__ == '__main__':

    lang = Language('en')
    text = [
        "google man",
        "man",
        "king",
        "woman",
        'queen',
        'man',
        'actor',
        'woman',
        'actress',
        'cat',
        'kitten',
        'puppy',
        'dog',
        'russia',
        'moscow',
        'france',
        'paris',
        'obama',
        'president',
        'trump',
        'executive',
        'rich',
        'mansion',
        'poor',
        'residence',
        'elvis',
        'rock',
        'eminem',
        'rap',
        'paper',
        'newspaper',
        'screen',
        'tv',
        'monet',
        'paint',
        'michelangelo',
        'leonardo',
        'beer',
        'barley',
        'wine',
        'rye',
        'earth',
        'moon',
        'sun',
        'house',
        'roof',
        'castle',
        'moat',
        'building',
        'architect',
        'software',
        'programmer',
        'boston',
        'bruins',
        'phoenix',
        'suns',
        'good',
        'heaven',
        'bad',
        'hell',
        'jordan',
        'basketball',
        'woods',
        'golf',
        'woman',
        'girl',
        'she',
        'teenager',
        'boy',
        'comedian',
        'actresses',
        'starred',
        'screenwriter',
        'puppy',
        'rottweiler',
        'puppies',
        'pooch',
        'pug']

    lang.build_pretrained_vocab(text, vectors='glove.6B.100d')

    print(len(lang.embed_word('<pad>')))

    def closest(vec, n=10):
        """
        Find the closest words for a given vector.

        :param vec: vector of an embedded word

        """
        all_dists = [(w, torch.dist(vec, lang.embed_word(w)))
                     for w in lang.vocab.itos]
        return sorted(all_dists, key=lambda t: t[1])[:n]

    def print_tuples(tuples):
        """
        Filters tuple so that it outputs (Euclidian distance) Word.

        :param tuples: list of tuples that contains euclidian distance and word string

        """

        for tuple in tuples:
            print('(%.4f) %s' % (tuple[1], tuple[0]))

    # In the form w1 : w2 :: w3 : ?
    def analogy(w1, w2, w3, n=5):
        """
        Finds the closest 5 words for vector operation w2.vector -w1.vector.

        +w3.vector where this function does the embedding.

        :param w1: String of word to be subtracted
        :param w2: String of word to be added
        :param w3: String of second word to be added
        :param n: number of words to search for
        :

        """
        print('\n[%s : %s :: %s : ?]' % (w1, w2, w3))

        # w2 - w1 + w3 = w4
        closest_words = closest(lang.embed_word(
            w2) - lang.embed_word(w1) + lang.embed_word(w3))

        closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]

        print_tuples(closest_words[:n])

    print_tuples(closest(lang.embed_word('google')))
    #analogy('man', 'king', 'woman')
    analogy('king', 'man', 'queen')
    analogy('man', 'actor', 'woman')

    index = lang.return_index_from_word('google')
    print("google's index is:", index)
    word = lang.return_word_from_index(index)
    print("Which as expected corresponds to:", word)

    print(lang.embed_sentence("Big Falcon Rocket is awesome").size())
    #analogy('cat', 'kitten', 'dog')
    #analogy('dog', 'puppy', 'cat')
    #analogy('russia', 'moscow', 'france')
    #analogy('obama', 'president', 'trump')
    #analogy('rich', 'mansion', 'poor')
    #analogy('elvis', 'rock', 'eminem')
    #analogy('paper', 'newspaper', 'screen')
    #analogy('monet', 'paint', 'michelangelo')
    #analogy('beer', 'barley', 'wine')
    # analogy('earth', 'moon', 'sun') # Interesting failure mode
    #analogy('house', 'roof', 'castle')
    #analogy('building', 'architect', 'software')
    #analogy('boston', 'bruins', 'phoenix')
    #analogy('good', 'heaven', 'bad')
    #analogy('jordan', 'basketball', 'woods')
