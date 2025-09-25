#!/usr/bin/env python
# coding: utf-8


### TENSORFLOW IMPLEMENTATION OF A DEEP LEARNING RECOMMENDATION SYSTEM 
# Alejandro Pineda, PhD

#%pip install tensorflow[and-cuda]


import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Set memory growth for the GPU
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
"""
import pandas as pd
import numpy as np
get_ipython().run_line_magic('pip', 'install pyspark')
import pyspark.sql.window as w
import pyspark.sql.functions as fn
from pyspark.sql.types import *
import random
#%pip install nltk
from nltk.tokenize import word_tokenize
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from statistics import mean
from datetime import datetime
from dateutil.relativedelta import relativedelta
import string 
import nltk
#nltk.download('punkt')
#nltk.download('punkt_tab')
from datetime import date, datetime
from statistics import mean
import kerastuner as kt
from tensorflow.keras import layers

path = ''
data = pd.read_csv(path)

# convert to pandas if using pyspark
#data = data.toPandas()
data.columns = data.columns.str.lower()

data.head(25)


data.columns


# In[3]:


df = data.drop(['original_contract_date', 'return_date', 'job_id', 'market_id', 'city_code', 'contract_key', 'contract_seq_key', 'state_code', 'location_id', 'district_id', 'inventory_region_id'], axis=1)


df_pandas = df
df_pandas.head()


# dealing with date/time constraints; comment out as necessary

"""
twelve_months_ago = datetime.now() - relativedelta(months=12)
twelve_months_ago


year_to_select = 2024
cutoff_date = '2025-01-01'

df_pandas['rental_start_dt'] = pd.to_datetime(df_pandas['rental_start_dt'])
df_pandas = df_pandas[df_pandas['rental_start_dt'] < cutoff_date]
df_pandas = df_pandas[df_pandas['RENTAL_START_DT'] >= twelve_months_ago]
df_pandas
"""


df_pandas.rental_start_dt.max()


len(df_pandas.item_id.unique())

# Use date.today() to establish training date
today_date = date.today()
today_date = today_date.strftime("%Y-%m-%d")
today_date


df_pandas['item_id'] = df_pandas['item_id'].str.replace('-', '', regex=False)
df_pandas.head(25)


df_pandas.dtypes


len(df_pandas.vertical.unique())


len(df_pandas.customer_id.unique())


vocab_size = len(df_pandas.item_id.unique()) + 1
print(vocab_size)


df_pandas = df_pandas.sort_values(by='rental_start_dt')
df_pandas


verts_unique_og = df_pandas.vertical.unique()
verts_unique_og


def strip_characters(text):
  """Removes numbers and symbols from a string, keeping only letters and spaces.

  Args:
    text: The input string.

  Returns:
    A new string with numbers and symbols removed.
  """
  return re.sub(r'[^a-zA-Z\s]', '', text)

def remove_punctuation(text):
    words = word_tokenize(text)
    punctuation_free = [word for word in words if word not in string.punctuation]
    return " ".join(punctuation_free)

df_pandas['vertical'] = df_pandas['vertical'].fillna('')
#df_pandas['vertical'] = df_pandas['vertical'].apply(remove_punctuation)
df_pandas['vertical'] = df_pandas['vertical'].str.lower()
df_pandas['vertical'] = df_pandas['vertical'].str.replace(' ', '', regex=False) 
df_pandas['vertical'] = df_pandas['vertical'].str.replace('-', '', regex=False)
df_pandas['vertical'] = df_pandas['vertical'].str.replace('&', '', regex=False)
df_pandas['vertical'] = df_pandas['vertical'].str.replace('(', '', regex=False)
df_pandas['vertical'] = df_pandas['vertical'].str.replace(')', '', regex=False)
df_pandas['vertical'] = df_pandas['vertical'].str.replace(',', '', regex=False)
df_pandas

# this code needs to be cleaned up^^^


vertical_vocab_size = len(df_pandas.vertical.unique())
vertical_vocab_size

verts_unique_cleaned = df_pandas.vertical.unique()
verts_unique_cleaned

vert_mapping = dict(zip(verts_unique_cleaned, verts_unique_og))
vert_mapping


# Combine rows based on 'col1' and 'col2', summing 'col3' and joining 'col4'
data_combo1 = df_pandas.groupby(['customer_id', 'rental_start_dt']).agg({'item_id': ', '.join,
                                                                     'vertical': ', '.join}).reset_index()
data_combo1.head(25)


#data_combo = data_combo1.groupby(['customer_id', 'vertical']).agg({'item_id': ', '.join}).reset_index()
#data_combo.head(25)


# downsample as required; this could be tied to computational limitations or just to speed up parameter tuning
data_combo = data_combo1.sample(frac=.05, random_state=1234)
data_combo


#seq_length = [len(i) for i in data_combo.sequences.values]
print("Average Length of sequences: ", mean(data_combo['item_id'].apply(len)))


data_combo.item_id


#verts = data_combo[['customer_id', 'vertical']]
#display(verts.head())


#vert_list = [[i[0] for i in x] for x in vertical]


#data_combo[data_combo['customer_id'] == 1189025]
len(data_combo.customer_id.unique())


import json

def write_dict_to_file(data, filename):
    """Writes a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to write.
        filename (str): The name of the file to write to.
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4) # indent makes the output more readable
        

### TOKENS: REPRESENTING WORDS WITH NUMBERS

# Create a tokenizer
tokenizer = Tokenizer()

# Fit the tokenizer on the text column
tokenizer.fit_on_texts(data_combo['vertical'])

# Convert text to sequences of integers
data_combo['vert_sequences'] = tokenizer.texts_to_sequences(data_combo['vertical'])

reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
#display(reverse_word_index)

write_dict_to_file(reverse_word_index, "reverse_word_index_vertical.json")

data_combo.head()


# Create another tokenizer

tokenizer = Tokenizer()

# Fit the tokenizer on the text column
tokenizer.fit_on_texts(data_combo['item_id'])

# Convert text to sequences of integers
data_combo['sequences'] = tokenizer.texts_to_sequences(data_combo['item_id'])

reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
#display(reverse_word_index)

write_dict_to_file(reverse_word_index, "reverse_word_index_catclass.json")

# Pad sequences to have the same length
#max_length = max(len(seq) for seq in data_combo['sequences'])
#data_combo['padded_sequences'] = pad_sequences(data_combo['sequences'], maxlen=max_length, padding='post').tolist()

data_combo.head()


print("Average Length of sequences: ", mean(data_combo['sequences'].apply(len)))


max_len = 12


# If you're only looking to grab unique values from the order history, try this; otherwise, comment out
#data_combo['seq_sets'] = [list(dict.fromkeys(i)) for i in data_combo.sequences] # use to grab sets of things
#data_combo['seq_sets']


# Specify the desired length
#desired_length = 10

# Filter the DataFrame
dat = data_combo[data_combo['sequences'].apply(len) >= 5]
dat


#sample_df = dat.sample(n=100000)
#sample_df.to_csv('customer_sample_qa_723.csv')


dat.shape


len(dat.customer_id.unique())


#1. DATA CLEAN UNTIL YOU HAVE THE FOLLOWING TENSOR...
# (once you have item_id, you also have target)
"""
train_dict = {
              'item_id': [[0, 0,...,0, 74, 276, 362], ...]
              'vertical':
              'target':  [[0, 0,...0, 276, 362, 119], ...]
             }
"""


len(dat.item_id.unique())


"""
data['class_name'] = data['class_name'].apply(remove_punctuation)
data['class_name'] = data['class_name'].str.lower()
data['class_name'] = data['class_name'].str.replace(' ', '', regex=False) 
data['class_name'] = data['class_name'].str.replace('-', '', regex=False) 
data['class_name'] = data['class_name'].str.replace('"', '', regex=False)
data['class_name'] = data['class_name'].str.replace("'", '', regex=False)
data['class_name'] = data['class_name'].str.replace('/', '', regex=False)
data.head(25)
"""


#data_combo = data.groupby(['customer_id', 'contract_id']).agg({'nb_days': ', '.join}).reset_index()
#data_combo.head(25)
#data_combo['order_length'] = [len(x) for x in data_combo['class_name']]
#data_combo


corpus = list(dat.item_id.unique())
num_words = len(corpus)
num_words



dat.shape


from tensorflow.keras.preprocessing.sequence import pad_sequences

og_sequences = list(dat.sequences.values)
og_sequences[0:1]


counter = 0
for i in og_sequences:
    if len(i) == 0:
        counter += 1

print(counter / len(og_sequences))


item_id = [x for x in dat.sequences.values]
#class_name = [(i.split(",")) for i in class_name]
item_id = [[int(i) for i in x] for x in item_id]
item_id = pad_sequences(item_id, maxlen=max_len, padding='pre', truncating='pre')
item_id[0:25]


#[ f(x) if condition else g(x) for x in sequence]
item_id = [i[:-1] for i in item_id]
item_id[0:1]


# the target sequences need the first element stripped...
target = [i[1:] for i in item_id]
target[0:1]


# zero pad to the 80th percentile of the sequence length
# make sure padding and truncating take place <post> sequence
item_id = pad_sequences(item_id, maxlen=max_len, padding='post', truncating='pre') 
item_id[0:1]


target = pad_sequences(target, maxlen=max_len, padding='post', truncating='pre')
target[0:1]


len(target)

dat.columns


vertical = [x for x in dat.vert_sequences.values]
#vertical = [[i + 100 for i in x] for x in vertical]
vertical = pad_sequences(vertical, maxlen=max_len, padding='pre', truncating='pre')
#vertical

train_dict = {
    'item_id': item_id, 
    'target': target,
    'vertical': vertical
             }
train_dict


# play with this tuning parameter to optimize training speed vs accuracy
batch_sz = 128


def create_train_tfdata(train_feat_dict, train_target_tensor,
                        batch_size, buffer_size=None):
    """
    Create train tf dataset for model train input
    :param train_feat_dict: dict, containing the features tensors for train data
    :param train_target_tensor: np.array(), the training TARGET tensor
    :param batch_size: (int) size of the batch to work with
    :param buffer_size: (int) Optional. Default is None. Size of the buffer
    :return: (tuple) 1st element is the training dataset,
                     2nd is the number of steps per epoch (based on batch size)
    """
    if buffer_size is None:
        buffer_size = batch_size*50

    train_steps_per_epoch = len(train_target_tensor) // batch_size

    train_dataset = tf.data.Dataset.from_tensor_slices((train_feat_dict,
                                                        train_target_tensor)).cache()
    for sample in train_dataset.take( 1 ):
        print("shape of X:", sample[ 0 ] )
        print("shape of y:", sample[ 1 ] )
        
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    train_dataset = train_dataset.repeat().prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_dataset, train_steps_per_epoch
  

item_id


# item_id: str, sequences of boughtt item ids for each customer
# nb_days: str, sequences of days from item purchase
train_feat_dict = {'item_id': item_id,
                     'vertical': vertical}
train_target_tensor = target
  
train_dataset, train_steps_per_epoch = create_train_tfdata(train_feat_dict,
                                                             train_target_tensor,
                                                             batch_size=batch_sz)
  


def loss_function(real, pred):
    """
    We redefine our own loss function in order to get rid of the '0' value
    which is the one used for padding. This to avoid that the model optimize itself
    by predicting this value because it is the padding one.
    
    :param real: the truth
    :param pred: predictions
    :return: a masked loss where '0' in real (due to padding)
                are not taken into account for the evaluation
    """

    # to check that pred is numric and not nan
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_object_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                 reduction='none')
    loss_ = loss_object_(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

train_steps_per_epoch

train_dataset


train_feat_dict['item_id']


# dictionary of hyperparameters; adjust number of units and learning rate to optimize training speed vs accuracy
hp_dict = {
    'embedding_item': max_len,
    'embedding_vert': max_len,
    'rnn_units_cat': 72,
    'learning_rate': .01
}


get_ipython().run_line_magic('pip', 'install keras-tuner')


# custom Keras layer to handle masking layer
# for more: https://www.tensorflow.org/guide/keras/understanding_masking_and_padding
from tensorflow.keras.layers import Layer


class LogicalNotLayer(Layer):
    def call(self, inputs):
        return tf.logical_not(tf.cast(inputs, tf.bool))


masking_layer = tf.keras.layers.Masking()

# try using bidirectional layers if accuracy gets stuck; this lets the model consider sequences backward and forward
#from tensorflow.keras import layers
#bidirectional_layer = layers.Bidirectional(layers.LSTM((max_len)))


#from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout, Bidirectional

def build_model(hp, max_len, item_vocab_size, vert_vocab_size):
    """
    Build a model given the hyper-parameters with item and nb_days input features
    :param hp: (kt.HyperParameters) hyper-parameters to use when building this model
    :return: built and compiled tensorflow model 
    """
    inputs = {}
    inputs['item_id'] = tf.keras.Input(shape=(max_len,), batch_size=batch_sz,
                                       name='item_id', dtype=tf.int32)
    # create encoding padding mask
    encoding_padding_mask = masking_layer(inputs['item_id'])
   
    # verticals
    inputs['vertical'] = tf.keras.Input(shape=(max_len,), batch_size=batch_sz,
                                       name='vertical', dtype=tf.int32)

    # Pass categorical input through embedding layer
    # with size equals to tokenizer vocabulary size
    # Remember that vocab_size is len of item tokenizer + 1
    # (for the padding '0' value)
    
    embedding_item = layers.Embedding(input_dim=item_vocab_size,
                                               output_dim=hp['embedding_item'],
                                               name='embedding_item'
                                              )(inputs['item_id'])
    # nbins=100, +1 for zero padding
    
    embedding_vertical = layers.Embedding(input_dim=vert_vocab_size,
                                               output_dim=hp['embedding_vert'],
                                               name='embedding_vertical'
                                              )(inputs['vertical'])
    

    #  Concatenate embedding layers
    concat_embedding_input = layers.Concatenate(name='concat_embedding_input')([embedding_item, embedding_vertical])

    concat_embedding_input = layers.BatchNormalization(name='batchnorm_inputs')(concat_embedding_input)
    
    #concat_embedding_input = tf.keras.layers.Reshape(target_shape=(12, 24))(concat_embedding_input)
    
    # LSTM layer
        
    rnn1 = layers.LSTM(units=hp['rnn_units_cat'],
                                   return_sequences=True,
                                   use_cudnn='auto',
                                   stateful=False,
                                   recurrent_initializer='orthogonal',
                                   name='LSTM_cat1'
                                   )(concat_embedding_input)

    rnn2 = layers.LSTM(units=hp['rnn_units_cat'],
                                   return_sequences=True,
                                   use_cudnn='auto',
                                   stateful=False,
                                   recurrent_initializer='orthogonal',
                                   name='LSTM_cat2'
                                   )(rnn1)

    rnn3 = layers.LSTM(units=hp['rnn_units_cat'],
                                   return_sequences=True,
                                   use_cudnn='auto',
                                   stateful=False,
                                   recurrent_initializer='orthogonal',
                                   name='LSTM_cat3'
                                   )(rnn2)

    rnn4 = layers.LSTM(units=hp['rnn_units_cat'],
                                   return_sequences=True,
                                   use_cudnn='auto',
                                   stateful=False,
                                   recurrent_initializer='orthogonal',
                                   name='LSTM_cat4'
                                   )(rnn3)

    rnn5 = layers.LSTM(units=hp['rnn_units_cat'],
                                   return_sequences=True,
                                   use_cudnn='auto',
                                   stateful=False,
                                   recurrent_initializer='orthogonal',
                                   name='LSTM_cat5'
                                   )(rnn4)

    rnn6 = layers.LSTM(units=hp['rnn_units_cat'],
                                   return_sequences=True,
                                   use_cudnn='auto',
                                   stateful=False,
                                   recurrent_initializer='orthogonal',
                                   name='LSTM_cat6'
                                   )(rnn5)

    rnn7 = layers.LSTM(units=hp['rnn_units_cat'],
                                   return_sequences=True,
                                   use_cudnn='auto',
                                   stateful=False,
                                   recurrent_initializer='orthogonal',
                                   name='LSTM_cat7'
                                   )(rnn6)

    rnn8 = layers.LSTM(units=hp['rnn_units_cat'],
                                   return_sequences=True,
                                   use_cudnn='auto',
                                   stateful=False,
                                   recurrent_initializer='orthogonal',
                                   name='LSTM_cat8'
                                   )(rnn7)


    batch = layers.BatchNormalization(name='batchnorm_lstm')(rnn8)

    # Self attention so key=value in inputs
    """
    att = layers.Attention(use_scale=False,
                                    name='attention')(inputs=[rnn6, rnn1],
                                                      mask=[encoding_padding_mask,
                                                            encoding_padding_mask],
                                                      use_causal_mask = True)
    
    """
    # Last layer is a fully connected one
        
    outputs = layers.Dense(item_vocab_size, name='output')(batch)

    model = tf.keras.Model(inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp['learning_rate']),
        loss=loss_function,
        metrics=['sparse_categorical_accuracy'])
    
    return model


rnn_mod0 = build_model(
    hp = hp_dict,
    max_len = max_len,
    item_vocab_size = vocab_size,
    vert_vocab_size = vertical_vocab_size
)


rnn_mod0


def fit_model(model, train_dataset, steps_per_epoch, epochs):
    """
    Fit the Keras model on the training dataset for a number of given epochs
    :param model: tf model to be trained
    :param train_dataset: (tf.data.Dataset object) the training dataset
                          used to fit the model
    :param steps_per_epoch: (int) Total number of steps (batches of samples) before 
                            declaring one epoch finished and starting the next epoch.
    :param epochs: (int) the number of epochs for the fitting phase
    :return: tuple (mirrored_model, history) with trained model and model history
    """
    
    # mirrored_strategy allows to use multi GPUs when available
    mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.AUTO)
    
    with mirrored_strategy.scope():
        mirrored_model = model

    history = mirrored_model.fit(train_dataset,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs, verbose=2)

    return mirrored_model, history


train_dataset

pip list | grep nvidia


"""
mod0 = rnn_mod0.fit(train_dataset,
          steps_per_epoch=train_steps_per_epoch,
          epochs=10, verbose=2)
"""


#import mlflow
#mlflow.autolog(disable=True)

mod0, history0 = fit_model(
    model= rnn_mod0,
    train_dataset= train_dataset,
    steps_per_epoch = train_steps_per_epoch,
    epochs = 500
          )


# use export over save so that the model can be called as an API endpoint
mod0.export('saved_model_60_8_500')


train_dataset


"""
Model requires three inputs:
item_id: shape=(None, 12)
nb_days: shape=(None, 12)
vertical: shape=(None, 12)
"""
mod0



item__id_test = train_dict['item_id'][0:63]
nb_days_test = train_dict['nb_days'][0:63]
vertical_test = train_dict['vertical'][0:63]


item__id_test


# Make prediction


prediction = mod0.predict({'item_id': item__id_test, 
                           'nb_days': nb_days_test,
                           'vertical': vertical_test})

# Grab the predicted classes
predicted_classes = np.argmax(prediction, axis=-1) # gives us the position of the largest weight (corresponds to our og item tokens)
print(predicted_classes)


predicted_classes

prediction[1]


import json

reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}
reverse_word_index

def write_dict_to_file(data, filename):
    """Writes a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to write.
        filename (str): The name of the file to write to.
    """
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4) # indent makes the output more readable

write_dict_to_file(reverse_word_index, "reverse_word_index.json")


# SAMPLE GRAB FOR TESTING
dat.shape



# saving a simple random sample for testing
sample_n = 10000
random_sample_df = dat.sample(n=sample_n, random_state=1234)
random_sample_df.to_csv('customer_sample_for_testing.csv', index=False)
#spark_df = spark.createDataFrame(random_sample_df)


purchase_recommendations = predicted_classes[1][0:3] # we want to iterate through and grab the first three of each


purchase_recommendations
print("Based on your purchase history, we recommend the following items: ")
for i in purchase_recommendations:
    print(reverse_word_index[i])



actuals = item__id_test[1][0:3]

for i in actuals:
    print(reverse_word_index[i])
