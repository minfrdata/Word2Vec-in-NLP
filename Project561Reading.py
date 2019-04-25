## -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:09:34 2017

@author: minxiaocn
"""

# ========================================================================
'''data manipulation'''
import os
filelink="/Users/minxiaocn/Desktop/20news-bydate/20news-bydate-train/"
import pandas as pd
import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disaable=redefined-builtin
import tensorflow as tf

import nltk
from nltk.tokenize import RegexpTokenizer


LOGDIR = "/tmp/mnist_tutorial/"

tokenizer = RegexpTokenizer(r'\w+')
#
#
##word list for an article
word=[]
category=[]
articles=[]
articles_untokenize=[]


os.chdir(filelink)

for filename in os.listdir(os.getcwd()):
    field=[]
   
    if filename !=".DS_Store":
        
        print(filename)
        os.chdir(filelink+filename)
        
        for txt in os.listdir(os.getcwd()):
            
            article=[]
            article_untokenize=""
            try:
                text=[]
                with open(txt,'r') as f:
                    for line in f.readlines():
                        
                        text.append(line.strip())
                for i in range(len(text)):
                    # remove the contents befor 1st space line
                    if (text[i]==''):
                        text=text[i+1:]
                        
                        break
                     
                for i in range(len(text)):
                    article_untokenize+=text[i]
                    words=tokenizer.tokenize(text[i])
                    article.extend(words)
                    word.extend(words)
                    field.extend(words) 
                articles.append(article)
                articles_untokenize.append(article_untokenize)
             
                category.append(filename)
                
            
            except:
                print("error file: "+txt)
        
        
        with open("/Users/minxiaocn/Desktop/window/filed_"+filename+".txt","w") as f:
            f.write("\n".join(field))

del article_untokenize
#build the numerical category 
category_dictionary={}            
combination=zip(range(len(set(category))),set(category))
for i,j in combination:
    category_dictionary[j]=i
numeri_category=list()
for i in category:
    index=category_dictionary.get(i,0)
    numeri_category.append(index)




word_lower=[l.lower() for l in word]
with open("/Users/minxiaocn/Desktop/window/fulltextdata_lower2.txt","w")as f:
    f.write("\n".join(word_lower))




## ========================================================================
## ========================================================================  
'''word2vec'''  
print("training begins")  
with open("/Users/minxiaocn/Desktop/window/fulltextdata_lower2.txt","r")as f:
    doc_lower=f.read().splitlines()
vocabulary=doc_lower
vocabulary_size =42634# keep words that occur 3 times or more



def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  print(count)
  print(collections.Counter(words).most_common(10))
  count.extend(collections.Counter(words).most_common(n_words - 1)) 
  # from the original data, we only keep n_words - 1 most common items and use them to
  # creat a dictionary
  
  
  dictionary = dict() # dictionary will be rank according to the word count number, count will only include the length of  vocaluary
  #size
  for word, _ in count:
    
    dictionary[word] = len(dictionary)
    
    
  data = list()# data is the numerical representation of the doc_lower
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0) 
    #if the word is not in the dictionary, it will return 0, so the UNK position will include all
    #the words count that is not in the dictionay
    #dictionary is the dictionary for source text, unique words, here we are producing value to text and text to value dictionary
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    data.append(index) # create a equivalent numeric file to represent original text
  count[0][1] = unk_count                # fill unk count to unk
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
  
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,vocabulary_size)

#only here
del vocabulary  # Hint to reduce memory.
#print('Most common words (+In)', count[:5])
#print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0
# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(data):
    data_index = 0
  buffer.extend(data[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(data):
      buffer[:] = data[:span]
      data_index = span
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(i)
    print(batch[i], reverse_dictionary[batch[i]],'->', labels[i, 0], reverse_dictionary[labels[i, 0]])



# Step 4: Build and train a skip-gram model.

batch_size = 100
embedding_size = 100 # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
num_sampled = 50    # Number of negative examples to sample.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)


graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # Explanation of the meaning of NCE loss:
  #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
  loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))
                     
#   with tf.name_scope("xent"):
#    xent = tf.reduce_mean(
#        tf.nn.softmax_cross_entropy_with_logits(
#            logits=logits, labels=y), name="xent")
#    tf.summary.scalar("xent", xent)
                     
                     
                       



   


  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)
      
      

 



  # Add variable initializer.
  init = tf.global_variables_initializer()
  
  
  

# Step 5: Begin training.
num_steps = 50000

with tf.Session(graph=graph) as session:
    
    
  
  tf.summary.scalar('nce_loss', loss)  
  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter("/tmp/mnist_logs"+ '/train',graph=tf.get_default_graph())
#  test_writer = tf.summary.FileWriter("/tmp/mnist_logs"+ '/test')
 
  
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val
    

    if step % 2000 == 0:
      summary=session.run(merged, feed_dict=feed_dict)
      train_writer.add_summary(summary, step)
        
      
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0
      
     

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in xrange(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in xrange(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        print(log_str)
  final_embeddings = normalized_embeddings.eval()
# Step 6: Visualize the embeddings.
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig("/Users/minxiaocn/Desktop/window/myplot.png")

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in xrange(plot_only)]
  plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)
  
t=pd.DataFrame(final_embeddings,index=reverse_dictionary.values())
t.to_csv("/Users/minxiaocn/Desktop/window/Final_embedding.csv")




#block to create doc embeddings

dictionary_embedding=pd.read_csv("/Users/minxiaocn/Desktop/window/Final_embedding.csv",index_col=0)
#

#import datasets and plot the matrix
#pcolor(dictionary_embedding.values)
#print(dictionary_embedding.values)


weighted_embedding=[]

for i in range(len(articles)):
    t=articles[i]
    if t==[]:
        del numeri_category[i]
        next
         
    else:
        tt=[]
        for i in t:
            if i  in dictionary.keys():
                tt.append(i)
            else:
                tt.append("UNK")
        if len(t)==1:
            numeric_article_t=dictionary_embedding.loc[tt[0],:]
            numeric_article_t_mean=numeric_article_t.tolist()
        else:
            numeric_article_t=dictionary_embedding.loc[tt,:]
            numeric_article_t_mean=numeric_article_t.mean(axis=0).tolist()
        
        weighted_embedding.append(numeric_article_t_mean)
        
weighted_embedding=pd.DataFrame(weighted_embedding)
weighted_embedding.to_csv("/Users/minxiaocn/Desktop/window/weighted_embedding.csv",index=False)
numeric_labels=pd.DataFrame(numeri_category)
numeric_labels.to_csv("/Users/minxiaocn/Desktop/window/numeric_labels.csv",index=False)






