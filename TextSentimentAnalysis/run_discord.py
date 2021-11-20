"""
    deploy as a discord bot
    make an aws EC2 instance and run this file on the instance - any other cloud service will do
    go to discord page and generate the bot and copy the bot token
"""

import nltk
nltk.download("punkt")
nltk.download('stopwords')

import argparse
import discord
import os
import dill
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

parser = argparse.ArgumentParser()
parser.add_argument('--token', type=str, help='discord bot token')

def load_model_vectorizer(pkl_file):
    with open(model_vectorizer_pkl, "rb") as pkl:
        return dill.load(pkl)

def prediction(sentence, vectorizer, model):
  #Apply preprocess steps
  lower_case = [word.lower() for word in sentence]
  cleaned_text = [word.translate(str.maketrans('','',string.punctuation)) for word in lower_case]
  tokenized_word = [word_tokenize(word) for word in cleaned_text]
  no_stop_word = [word for word in tokenized_word[0] if word not in stopwords.words("english")]

  #Vectorize the words
  vec_text = vectorizer.transform([no_stop_word])

  # #Prediction
  predicted = model.predict(vec_text)
  if predicted == 1:
      output = "sentece {} is a negative comment!".format(sentence)
  elif predicted == 2:
      output = "sentece {} is a positive comment!".format(sentence)
  else:
      output = "sentece {} is a neutral comment!".format(sentence)

  return output

def discord_bot(model_vectorizer_pkl,token):
    # load in the model
    vectorizer, model = load_model_vectorizer(model_vectorizer_pkl)

    # discord bot client
    client = discord.Client()

    @client.event
    async def on_ready():
       print('We have logged in as {0.user}'.format(client))

    @client.event
    async def on_message(message):
       if message.author == client.user:
           return

       sentence = message.content
       respond = prediction([sentence], vectorizer, model)
       await message.channel.send(respond)

    # discord bot token
    client.run(token)

if __name__ == '__main__':
    args = parser.parse_args()
    model_vectorizer_pkl = 'text_sentiment.pkl'
    discord_bot(model_vectorizer_pkl, args.token)
