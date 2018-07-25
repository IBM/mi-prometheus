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
                Pretrained embedding handler inspired by vocab.py in torchtext
__author__ = "Vincent Marois, Ryan L. McAvoy"
"""

import torch
import torchtext.vocab as vocab

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
        self.word_embedding = vocab.Glove(name='6B', dim= 100)
        self.n_words = 3  # Number of words in the language. Start by counting PAD, EOS, SOS tokens.
        self.embedding_size = 1

        if (pretrained):
           self.load_pretrained
           

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
        outsentence = torch.zeros((sentence.size()[0], self.word_embedding.size()[1]
        for i, word in enumerate(sentence):
            outsentence[i,:] = self.embed_word(word)
            
        return outsentence

        #return self.embedding(sentence)
        #currently just does a dummy reshape and returns LongTensor of size [max_sentence_length, 1]
        return torch.unsqueeze(sentence,1)
    
    def embed_word(self, word):
        index=self.word_embedding.stoi[word]
        return self.word_embedding.vectors[index]
       
     


class WordEmbedding:
    def __init__(self, embed_file, url=None, data_dir= ".embedding dir", unknown_init=torch.Tensor.zero_):
       """
       Class for reading embedding vectors from file or downloading them
           :param embed_file: The name of the file that contains the pretrained embedding
           :param url: The URL of the embedding file
           :param data_dir: The directory that will store the embeddings or already does
           :unknown_init: The function that initializes the out of vocab word vectors. Takes a vector returns a vector of the same size
       """
       self.unknown_init = unknown_init
       self.load_vectors(embed_file, url, data_dir)

    def load_wordembedding(self, embed_file, data_dir, url):
        
        raw_dir = os.path.join(data_dir, "raw")
        processed_dir = os.path.join(data_dir, "processed")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            os.makedirs(raw_dir)
            os.makedirs(processed_dir)

        convert_embed = os.path.join(processed_dir, os.path.basename(embed_file)) + '.pt'
        original_embed = os.path.join(raw_dir, os.path.basename(embed_file))
        
        #TODO Check if valid URL
        if not os.path.isfile(path_save):
            if not os.path.isfile(embed_file) and url:
                self.download_wordembedding(raw_dir,  url)
            else:
                #No plans to use this option at the minute
                original_embed = embed_file

            self.convert_wordembedding(original_embed, convert_embed)
                    
        else:
            logger.info('Loading vectors from {}'.format(convert_embed))
            self.index2word, self.word2index, self.word_embedding, self.dim = torch.load(path_pt)

    def download_wordembedding(data_dir, url):
        import httplib
        c = httplib.HTTPConnection(url)
        c.request("HEAD", '')
        if c.getresponse().status == 200:
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
        else:
            logger.error('Website {} does not exist'.format(url))

    def convert_wordembedding(original_embed, convert_embed):
       
        itos  = []
        vectors =[]
        dim = None

        
        logger.info("Loading vectors from {}".format(orignal))
        with io.open(original_embed, encoding="utf8") as f:
             lines = [line for line in f]
        
        for line in lines:
                line_split = line.rstrip().split(" ")

                word, vector = line_split[0], line_split[1:]
                if dim is None and len(vector) > 1:
                    dim = len(entries)
                elif len(entries) <= 1:
                    logger.warning("Skipping token {} with 0 or 1 dim "
                                   "vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):
                    raise RuntimeError(
                        "Vectors are not all of the same size. Token {} has {} dimensions, but we expected dim {}.".format(word, len(entries), dim))

                vectors.extend(float(x) for x in vector)
                itos.append(word)

              self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.word_embedding = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(convert_embed))
            torch.save((self.itos, self.stoi, self.vectors, self.dim), convert_embed)



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

print_tuples(closest(get_word('google')))

def closest(vec, n=10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in glove.itos]
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

analogy('king', 'man', 'queen')
analogy('man', 'actor', 'woman')
analogy('cat', 'kitten', 'dog')
analogy('dog', 'puppy', 'cat')
analogy('russia', 'moscow', 'france')
analogy('obama', 'president', 'trump')
analogy('rich', 'mansion', 'poor')
analogy('elvis', 'rock', 'eminem')
analogy('paper', 'newspaper', 'screen')
analogy('monet', 'paint', 'michelangelo')
analogy('beer', 'barley', 'wine')
analogy('earth', 'moon', 'sun') # Interesting failure mode
analogy('house', 'roof', 'castle')
analogy('building', 'architect', 'software')
analogy('boston', 'bruins', 'phoenix')
analogy('good', 'heaven', 'bad')
analogy('jordan', 'basketball', 'woods')
