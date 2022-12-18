import json
import requests
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')

model=load_model('chatbotdata.h5')

data = {"intents": [
             {"tag": "greeting",
              "patterns": ["Hello", "Hi", "Apa Khabar?", "Salam", "Selamat pagi", "Selamat petang", "Selamat mlm"],
              "responses": ["Hi", "Hello"],
             },
             {"tag": "age",
              "patterns": ["Umur berapa?", "Bila birthday?", "Bila wujud?", "Umur brape?", "Bila lahir?", "Tarikh lahir bila?", "Tarikh birthday bila?"],
              "responses": ["Umur saya 3bulan", "Saya lahir pada 2022", "Saya diwujudkan pada 7 Januari 2022", "07/01/2022"]
             },            
             {"tag": "master",
              "patterns": ["Sapa tuan awak?", "Sapa buat awak?", "Awak milik siapa?"],
              "responses": ["Saya adalah kepunyaan FSKM UiTM Shah Alam"]
             },
             {"tag": "goodbye",
              "patterns": [ "bye", "see ya", "Gerak lu", "Cau2", "Cau", "Saya pergi dulu", "Thank you", "Terima Kasih"],
              "responses": ["Bye", "Jumpa lagi", "Semoga kita berurusan lagi!"]
             }
]}

# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()
# Each list to create
words = []
classes = []
doc_X = []
doc_y = []
# Loop through all the intents
# tokenize each pattern and append tokens to words, the patterns and
# the associated tag to their associated list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])
    
    # add the tag to the classes if it's not there already 
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))


def clean_text(text): 
  tokens = nltk.word_tokenize(text.lower())
  tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
  return tokens

def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result


token = "5277992227:AAHXpye9cyJQh72G-93Q10uZbui7kZuEDIs"
import os
from flask import Flask, request
import telebot
from telebot import *
bot = telebot.TeleBot(token)
server = Flask(__name__)


@bot.message_handler(commands=['start', 'help', 'mula', 'tolong'])
def send_welcome(message):
	bot.reply_to(message, "Hi, Saya Alumni-bot dan saya akan menolong anda berkenaan dengan persoalan tentang Alumni seperti berkenaan persatuan Alumni, Tabung sumbangan fskm, Skim Khidmat Pelajar(SKP), Rekod Pelajar dan Graduasi",  reply_markup=start_keyboard())



def start_keyboard():
    return types.InlineKeyboardMarkup(
        keyboard=[
            [
                types.InlineKeyboardButton(
                    text='Tanya Soalan',
                    callback_data='tanya'
                )
            ]
        ]
    )

@bot.callback_query_handler(func=lambda c: c.data == 'tanya')
def back_callback(call: types.CallbackQuery):
    bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                          text='Awak boleh bertanya berkenaan alumni')               

@bot.message_handler(commands=['tanya'])
def send_ask(message):
  bot.reply_to(message, "Awak boleh mula bertanya")


@bot.message_handler(content_types=['text'])
def message_received(message):
  intents = pred_class(message.text, words, classes)
  result = get_response(intents, data)
  bot.send_message(chat_id=message.from_user.id, text=result)



bot.polling(True)