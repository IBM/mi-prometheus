#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

#import torchtext.vocab as vocab
from __future__ import unicode_literals
import array
from collections import defaultdict
from functools import partial
import io
import logging
import os
import zipfile

import six
from six.moves.urllib.request import urlretrieve
import torch
from tqdm import tqdm
import tarfile
from collections import Counter, OrderedDict

logger = logging.getLogger(__name__)


class Lang:
    """Simple helper class allowing to represent a language in a translation task. It will contain for instance a vocabulary
    index (word2index dict) & keep track of the number of words in the language.

    The inputs and targets of the associated sequence to sequence networks will be sequences of indexes, each item
    representing a word. The attributes of this class (word2index, index2word, word2count) are useful to keep track of
    this.
    """

    def __init__(self, name):
        """
        Constructor.
        :param name: string to name the language (e.g. french, english)
        """
        self.name = name
        #self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2}  # dict 'word': index
        #self.word2count = {}  # keep track of the occurrence of each word in the language. Can be used to replace
        # rare words.
        #self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}  # dict 'index': 'word', initializes with PAD, EOS, SOS tokens
        self.vocab_cls = Vocab
        self.init_token = None
        self.eos_token = None
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"

       #if (pretrained):
        #self.build_vocab(data_set)
           

    def add_sentence(self, sentence):
        """
        Process a sentence using add_word()
        :param sentence: sentence to be added to the language
        :return: None
        """
        for word in sentence.split(' '):
            self.add_word(word)
   
    def add_word(self, word):
        """
        Add a word to the vocabulary set: update word2index, word2count, index2words & n_words.
        :param word: word to be added.
        :return: None.
        """

        if word not in self.word2index:  # if the current word has not been seen before
            self.word2index[word] = self.n_words  # create a new entry in word2index
            self.word2count[word] = 1  # count first occurrence of this word
            self.index2word[self.n_words] = word  # create a new entry in index2word
            self.n_words += 1  # increment total number of words in the language

        else:  # this word has been seen before, simply update its occurrence
            self.word2count[word] += 1

 
    def embed_sentence(self, sentence):
        """
        Embed an entire sentence using a pretrained embedding
        :param sentence: A pytorch LongTensor of word indices [max_sentence_length]
        :returns: FloatTensor of embedded vectors [max_sentence_length, embedding size]
        """
        outsentence = torch.zeros((len(sentence.split()), self.vocab.vectors.size()[1]))
        for i, word in enumerate(sentence.split()):
            outsentence[i,:] = self.embed_word(word)
            
        return outsentence

        #return self.embedding(sentence)
        #currently just does a dummy reshape and returns LongTensor of size [max_sentence_length, 1]
        return torch.unsqueeze(sentence,1)
    
    def embed_word(self, word):
        index=self.vocab.stoi[word]
        return self.vocab.vectors[index]
       
     

    def build_pretrained_vocab(self, data_set, **kwargs):
        """Construct the Vocab object for this field from a list of sentences.
        Arguments:
            Positional arguments:          
            Remaining keyword arguments: Passed to the constructor of Vocab.
        """

        counter = Counter()
        #Break list of sentences into sentences
        for data in data_set:
            counter.update(data.split())

        specials = list(OrderedDict.fromkeys(
            tok for tok in [self.unk_token, self.pad_token, self.init_token,
                            self.eos_token]
            if tok is not None))
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.stoi = defaultdict(_default_unk_index)
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unknown_init=unk_init, data_dir=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

    def load_vectors(self, vectors, **kwargs):
        """
        Arguments:
            vectors: one of or a list containing instantiations of the
                GloVe, CharNGram, or Vectors classes. Alternatively, one
                of or a list of available pretrained vectors:
                charngram.100d
                fasttext.en.300d
                fasttext.simple.300d
                glove.42B.300d
                glove.840B.300d
                glove.twitter.27B.25d
                glove.twitter.27B.50d
                glove.twitter.27B.100d
                glove.twitter.27B.200d
                glove.6B.50d
                glove.6B.100d
                glove.6B.200d
                glove.6B.300d
            Remaining keyword arguments: Passed to the constructor of Vectors classes.
        """
        if not isinstance(vectors, list):
            vectors = [vectors]
        for idx, vector in enumerate(vectors):
            if six.PY2 and isinstance(vector, str):
                vector = six.text_type(vector)
            if isinstance(vector, six.string_types):
                # Convert the string pretrained vector identifier
                # to a Vectors object
                if vector not in pretrained_aliases:
                    raise ValueError(
                        "Got string input vector {}, but allowed pretrained "
                        "vectors are {}".format(
                            vector, list(pretrained_aliases.keys())))
                vectors[idx] = pretrained_aliases[vector](**kwargs)
            elif not isinstance(vector, Vectors):
                raise ValueError(
                    "Got input vectors of type {}, expected str or "
                    "Vectors object".format(type(vector)))

        tot_dim = sum(v.dim for v in vectors)
        self.vectors = torch.Tensor(len(self), tot_dim)
        for i, token in enumerate(self.itos):
            start_dim = 0
            for v in vectors:
                end_dim = start_dim + v.dim
                self.vectors[i][start_dim:end_dim] = v[token.strip()]
                start_dim = end_dim
            assert(start_dim == tot_dim)

    def set_vectors(self, stoi, vectors, dim, unk_init=torch.Tensor.zero_):
        """
        Set the vectors for the Vocab instance from a collection of Tensors.
        Arguments:
            stoi: A dictionary of string to the index of the associated vector
                in the `vectors` input argument.
            vectors: An indexed iterable (or other structure supporting __getitem__) that
                given an input index, returns a FloatTensor representing the vector
                for the token associated with the index. For example,
                vector[stoi["string"]] should return the vector for "string".
            dim: The dimensionality of the vectors.
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_
        """
        self.vectors = torch.Tensor(len(self), dim)
        for i, token in enumerate(self.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                self.vectors[i] = vectors[wv_index]
            else:
                self.vectors[i] = unk_init(self.vectors[i])


"""
FIGURE OUT WHETHER I NEED THIS EXTRA LICENSE SINCE I BASED IT ON THEIR CODE 
BSD 3-Clause License

Copyright (c) James Bradbury and Soumith Chintala 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

class WordEmbedding(object):
    def __init__(self, embed_file, url=None, data_dir= None, unknown_init=None):
       """
       Class for reading embedding vectors from file or downloading them
           :param embed_file: The name of the file that contains the pretrained embedding
           :param url: The URL of the embedding file
           :param data_dir: The directory that will store the embeddings or already does
           :unknown_init: The function that initializes the out of vocab word vectors. Takes a vector returns a vector of the same size
       """
       print(data_dir)
       data_dir = '.embedding_dir' if data_dir is None else data_dir
       self.unknown_init = torch.Tensor.zero_ if unknown_init is None else unknown_init
       self.load_wordembedding(embed_file, data_dir, url)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unknown_init(torch.Tensor(1, self.dim))

    def load_wordembedding(self, embed_file, data_dir, url):
       
        print(embed_file)
        print(data_dir)
        print(url) 
        raw_dir = os.path.join(data_dir, "raw")
        processed_dir = os.path.join(data_dir, "processed")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            os.makedirs(raw_dir)
            os.makedirs(processed_dir)

        convert_embed = os.path.join(processed_dir, os.path.basename(embed_file)) + '.pt'
        original_embed = os.path.join(raw_dir, os.path.basename(embed_file))
        
        #TODO Check if valid URL
        if not os.path.isfile(convert_embed):
            if not os.path.isfile(embed_file) and url:
                self.download_wordembedding(raw_dir,  url)
            else:
                #No plans to use this option at the minute
                original_embed = embed_file

            self.convert_wordembedding(original_embed, convert_embed)
                    
        else:
            logger.info('Loading vectors from {}'.format(convert_embed))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(convert_embed)

    def download_wordembedding(self, data_dir, url):
        #import http.client
        from six.moves.urllib.request import Request, urlopen
        print(data_dir)
        print(url)
        #c = http.client.HTTPConnection(url)
        #c.request("HEAD", '')
        #if c.getresponse().status == 200:
        logger.info('Downloading original source file from {}'.format(url))
        download_dest = os.path.join(data_dir, os.path.basename(url))
         
        # have to do a Request in order to pass headers to avoid server security features blocking spider/bot user agent
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'})
        data = urlopen(request)

        with open(download_dest, 'wb') as f:
            f.write(data.read())

        logger.info('Extracting original word embedding into {}'.format(data_dir))
        ext = os.path.splitext(download_dest)[1][1:]
        if ext == 'zip':
            with zipfile.ZipFile(download_dest, "r") as zf:
                zf.extractall(data_dir)
        elif ext == 'gz':
            with tarfile.open(download_dest, 'r:gz') as tar:
                tar.extractall(path=data_dir)
        #else:
        #    logger.error('Website {} does not exist'.format(url))

    def convert_wordembedding(self, original_embed, convert_embed):
       
        itos  = []
        vectors =[]
        dim = None

        
        logger.info("Loading vectors from {}".format(original_embed))
        with io.open(original_embed, encoding="utf8") as f:
             lines = [line for line in f]
        
        for line in lines:
            line_split = line.rstrip().split(" ")

            word, vector = line_split[0], line_split[1:]
            if dim is None and len(vector) > 1:
                dim = len(vector)
            elif len(vector) <= 1:
                logger.warning("Skipping token {} with 0 or 1 dim "
                               "vector {}; likely a header".format(word, vector))
                continue
            elif dim != len(vector):
                raise RuntimeError(
                    "Vectors are not all of the same size. Token {} has {} dimensions, but we expected dim {}.".format(word, len(vector), dim))

            vectors.extend(float(x) for x in vector)
            itos.append(word)

        self.itos = itos
        self.stoi = {word: i for i, word in enumerate(itos)}
        self.vectors = torch.Tensor(vectors).view(-1, dim)
        self.dim = dim
        logger.info('Saving vectors to {}'.format(convert_embed))
        torch.save((self.itos, self.stoi, self.vectors, self.dim), convert_embed)


class GloVe(WordEmbedding):
    url = {
        '42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        '840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        '6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
    }

    def __init__(self, name='840B', dim=300, **kwargs):
        url = self.url[name]
        name = 'glove.{}.{}d.txt'.format(name, str(dim))
        super(GloVe, self).__init__(name, url=url, **kwargs)


class FastText(WordEmbedding):

    url_base = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.{}.vec'

    def __init__(self, language="en", **kwargs):
        url = self.url_base.format(language)
        name = os.path.basename(url)
        super(FastText, self).__init__(name, url=url, **kwargs)


class CharNGram(WordEmbedding):

    name = 'charNgram.txt'
    url = ('http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/'
           'jmt_pre-trained_embeddings.tar.gz')

    def __init__(self, **kwargs):
        super(CharNGram, self).__init__(self.name, url=self.url, **kwargs)

    def __getitem__(self, token):
        vector = torch.Tensor(1, self.dim).zero_()
        if token == "<unk>":
            return self.unk_init(vector)
        # These literals need to be coerced to unicode for Python 2 compatibility
        # when we try to join them with read ngrams from the files.
        chars = ['#BEGIN#'] + list(token) + ['#END#']
        num_vectors = 0
        for n in [2, 3, 4]:
            end = len(chars) - n + 1
            grams = [chars[i:(i + n)] for i in range(end)]
            for gram in grams:
                gram_key = '{}gram-{}'.format(n, ''.join(gram))
                if gram_key in self.stoi:
                    vector += self.vectors[self.stoi[gram_key]]
                    num_vectors += 1
        if num_vectors > 0:
            vector /= num_vectors
        else:
            vector = self.unk_init(vector)
        return vector


def _default_unk_index():
    return 0


pretrained_aliases = {
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
}
"""Mapping from string name to factory function"""


#unit test of vocab
#pretrained embedding
#Based on the example of 
#https://github.com/spro/practical-pytorch/blob/master/glove-word-vectors/glove-word-vectors.ipynb
"""
The MIT License (MIT)

Copyright (c) 2017 Sean Robertson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
if __name__ == '__main__':



    lang = Lang('en')
    text=["google man", "man", "king", "woman", 'queen','man','actor','woman','actress','cat','kitten', 'puppy','dog','russia','moscow','france','paris','obama','president',
          'trump', 'executive', 'rich', 'mansion', 'poor', 'residence', 'elvis', 'rock', 'eminem', 'rap','paper','newspaper','screen','tv','monet','paint','michelangelo','leonardo',
          'beer', 'barley', 'wine','rye', 'earth','moon','sun', 'house', 'roof', 'castle', 'moat', 'building', 'architect','software','programmer','boston','bruins','phoenix','suns',
          'good', 'heaven', 'bad','hell','jordan','basketball','woods','golf', 'woman', 'girl','she','teenager', 'boy','comedian','actresses','starred','screenwriter','puppy','rottweiler', 'puppies','pooch','pug']


    lang.build_pretrained_vocab(text,vectors='glove.6B.100d')

    #print(lang.embed_word('<pad>'))

    def embed_sentence(sentence):
        return lang.embed_sentence(sentence)

    def get_word(word):
        return lang.embed_word(word)

    def closest(vec, n=10):
        """
        Find the closest words for a given vector
        """
        all_dists = [(w, torch.dist(vec, get_word(w))) for w in lang.vocab.itos]
        return sorted(all_dists, key=lambda t: t[1])[:n]

    def print_tuples(tuples):
        for tuple in tuples:
            print('(%.4f) %s' % (tuple[1], tuple[0]))

    # In the form w1 : w2 :: w3 : ?
    def analogy(w1, w2, w3, n=5, filter_given=True):
        print('\n[%s : %s :: %s : ?]' % (w1, w2, w3))

        # w2 - w1 + w3 = w4
        closest_words = closest(get_word(w2) - get_word(w1) + get_word(w3))

        # Optionally filter out given words
        if filter_given:
            closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]

        print_tuples(closest_words[:n])

    print_tuples(closest(get_word('google')))
    #analogy('man', 'king', 'woman')
    analogy('king', 'man', 'queen')
    analogy('man', 'actor', 'woman')

    print(embed_sentence("Big Falcon Rocket is awesome").size())
    #analogy('cat', 'kitten', 'dog')
    #analogy('dog', 'puppy', 'cat')
    #analogy('russia', 'moscow', 'france')
    #analogy('obama', 'president', 'trump')
    #analogy('rich', 'mansion', 'poor')
    #analogy('elvis', 'rock', 'eminem')
    #analogy('paper', 'newspaper', 'screen')
    #analogy('monet', 'paint', 'michelangelo')
    #analogy('beer', 'barley', 'wine')
    #analogy('earth', 'moon', 'sun') # Interesting failure mode
    #analogy('house', 'roof', 'castle')
    #analogy('building', 'architect', 'software')
    #analogy('boston', 'bruins', 'phoenix')
    #analogy('good', 'heaven', 'bad')
    #analogy('jordan', 'basketball', 'woods')
