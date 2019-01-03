# -*- coding: utf-8 -*-

import tensorflow as tf

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cut_word", "./data/vector/corpus_seg.txt", "训练的分词语料目录")
tf.app.flags.DEFINE_string("model_dir", "./corpus.model", "训练之后词向量的模型目录")


if __name__=='__main__':

  model = Word2Vec(LineSentence(FLAGS.cut_word), size=400, window=5, min_count=5)

  model.save(FLAGS.model_dir)
