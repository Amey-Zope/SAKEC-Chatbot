import random
import json
import pickle # for serialization
import numpy as np

import sys

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import tkinter as tk
from functools import partial

from tkinter import *

# Top level window
root = tk.Tk()
root.title("SAKEC Bot")
root.geometry('590x600')




main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)


my_canvas = Canvas(main_frame)
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)


my_scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)


my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))



second_frame = Frame(my_canvas)

my_canvas.create_window((0, 0), window=second_frame, anchor="nw")

















# frame=Frame(root)
# frame.pack(expand=True, fill=BOTH) #.grid(row=0,column=0)


# canvas=Canvas(frame,bg='#FFFFFF')
# hbar=Scrollbar(frame,orient=HORIZONTAL)
# hbar.pack(side=BOTTOM,fill=X)
# hbar.config(command=canvas.xview)

# vbar=Scrollbar(frame,orient=VERTICAL)

# vbar.pack(side=RIGHT,fill=Y)
# vbar.config(command=canvas.yview)
# canvas.config(width=3000,height=3000)
# canvas.config(yscrollcommand=vbar.set)
# canvas.pack(side=LEFT, expand=True, fill=BOTH)





# frame=Frame(root,width=3000,height=3000)
# frame.grid(row=0,column=0)




# canvas = tk.Canvas(frame, bg='#FFFFFF', width=3000, height=3000)


# vbar=Scrollbar(frame,orient=VERTICAL)
# vbar.grid(row=0, column=3, sticky=N+S+W)

# vbar.config(command=canvas.yview)

# canvas.config(yscrollcommand=vbar.set)

# canvas.grid(row=0, column=0, sticky="news")


bot_intro_lbl_en = tk.Label(second_frame, text = "Ask me about SAKEC", font=('Helvatical bold',10))
bot_intro_lbl_en.grid(column=1, row=0)
bot_intro_lbl_hi = tk.Label(second_frame, text = "SAKEC के बारे में मुझसे पूछें", font=('Helvatical bold',10))
bot_intro_lbl_hi.grid(column=1, row=1)
bot_intro_lbl_mr = tk.Label(second_frame, text = "SAKEC बद्दल मला विचारा", font=('Helvatical bold',10))
bot_intro_lbl_mr.grid(column=1, row=2)
bot_intro_lbl_gi = tk.Label(second_frame, text = "મને SAKEC વિશે પૂછો", font=('Helvatical bold',10))
bot_intro_lbl_gi.grid(column=1, row=3)


# Function for getting Input
# from textbox and printing it 
# at label widget
  

# method to print bot response
# def printBotRes():
#     inp = inputtxt.get(1.0, "end-1c")
#     lbl.config(text = "Provided Input: "+inp)





lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', encoding='utf8').read()) # to get JSON object in intents variable

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('SAKEC-Chatbot-model.h5')


# for cleaning up the input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# for converting sentence into a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if(word == w):
                bag[i] = 1

    return np.array(bag)

# to predict class of the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25 # 25% uncertainty is allowed
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result













# devadict= {
# 'u0900':' ऀ',
# 'u0901':' ँ',
# 'u0902':' ं',
# 'u0903': 'ः',
# 'u0904': 'ऄ',
# 'u0905': 'अ',
# 'u0906': 'आ',
# 'u0907': 'इ',
# 'u0908': 'ई',
# 'u0909': 'उ',
# 'u090A': 'ऊ',
# 'u090B': 'ऋ',
# 'u090C': 'ऌ',
# 'u090D': 'ऍ',
# 'u090E': 'ऎ',
# 'u090F': 'ए',
# 'u0910': 'ऐ',
# 'u0911': 'ऑ',
# 'u0912': 'ऒ',
# 'u0913': 'ओ',
# 'u0914': 'औ',
# 'u0915': 'क',
# 'u0916': 'ख',
# 'u0917': 'ग',
# 'u0918': 'घ',
# 'u0919': 'ङ',
# 'u091A': 'च',
# 'u091B': 'छ',
# 'u091C': 'ज',
# 'u091D': 'झ',
# 'u091E': 'ञ',
# 'u091F': 'ट',
# 'u0920': 'ठ',
# 'u0921': 'ड',
# 'u0922': 'ढ',
# 'u0923': 'ण',
# 'u0924': 'त',
# 'u0925': 'थ',
# 'u0926': 'द',
# 'u0927': 'ध',
# 'u0928': 'न',
# 'u0929': 'ऩ',
# 'u092A': 'प',
# 'u092B': 'फ',
# 'u092C': 'ब',
# 'u092D': 'भ',
# 'u092E': 'म',
# 'u092F': 'य',
# 'u0930': 'र',
# 'u0931': 'ऱ',
# 'u0932': 'ल',
# 'u0933': 'ळ',
# 'u0934': 'ऴ',
# 'u0935': 'व',
# 'u0936': 'श',
# 'u0937': 'ष',
# 'u0938': 'स',
# 'u0939': 'ह',
# 'u093A': 'ऺ',
# 'u093B': 'ऻ',
# 'u093C':' ़' ,
# 'u093D': 'ऽ',
# 'u093E': 'ा',
# 'u093F': 'ि',
# 'u0940': 'ी',
# 'u0941':' ु',
# 'u0942':' ू',
# 'u0943':' ृ',
# 'u0944':' ॄ',
# 'u0945':' ॅ',
# 'u0946':' ॆ',
# 'u0947':' े',
# 'u0948':' ै',
# 'u0949': 'ॉ',
# 'u094A': 'ॊ',
# 'u094B': 'ो',
# 'u094C': 'ौ',
# 'u094D':' ्',
# 'u094E': 'ॎ',
# 'u094F': 'ॏ',
# 'u0950': 'ॐ',
# 'u0951':'',
# 'u0952': '-',
# 'u0953':' ॓',
# 'u0954':' ॔',
# 'u0955':' ॕ',
# 'u0956': 'ॖ',
# 'u0957': 'ॗ',
# 'u0958': 'क़',
# 'u0959': 'ख़ ',
# 'u095A': 'ग़ ',
# 'u095B': 'ज़' ,
# 'u095C': 'ड़' ,
# 'u095D': 'ढ़' ,
# 'u095E': 'फ़' ,
# 'u095F': 'य़',
# 'u0960': 'ॠ',
# 'u0961': 'ॡ' ,
# 'u0962':' ॢ' ,
# 'u0963':' ॣ',
# 'u0964': '।' ,
# 'u0965': '॥' ,
# 'u0966': '०' ,
# 'u0967': '१' ,
# 'u0968': '२' ,
# 'u0969': '३' ,
# 'u096A': '४' ,
# 'u096B': '५' ,
# 'u096C': '६' ,
# 'u096D': '७' ,
# 'u096E': '८' ,
# 'u096F': '९',
# 'u0970': '॰',
# 'u0971': 'ॱ ',
# 'u0972': 'ॲ' ,
# 'u0973': 'ॳ' ,
# 'u0974': 'ॴ' ,
# 'u0975': 'ॵ' ,
# 'u0976': 'ॶ' ,
# 'u0977': 'ॷ' ,
# 'u0978': 'ॸ' ,
# 'u0979': 'ॹ' ,
# 'u097A': 'ॺ' ,
# 'u097B': 'ॻ' ,
# 'u097C': 'ॼ' ,
# 'u097D': 'ॽ' ,
# 'u097E': 'ॾ' ,
# 'u097F': 'ॿ',
# }

class ChatApp:

    def get_bot_res(self, inputtxt, row_num):

        user_res_list.append(inputtxt.get(1.0, "end-1c"))

        user_ip = user_res_list[-1]

        print("Current msg: ", inputtxt.get(1.0, "end-1c"))

        # botLabel = tk.Label(second_frame, text = "Sakecu > ")
        # botLabel.grid(column=0, row=row_num)

        lbl = tk.Label(second_frame, text = "\n", font=('Helvatical bold', 10), anchor = "w")
        lbl.grid(column=1, row=row_num, sticky="W")
        # lbl.pack()




        # ip_msg = input("You > ")
        ints = predict_class(user_ip)

        print("ints: ", ints)
              
        
        res = get_response(ints, intents)

        # lbl.config(text = res)


        my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))
        my_scrollbar.configure(command=my_canvas.yview)
        my_scrollbar.pack(side=RIGHT, fill=Y)

        if(ints[0]["intent"] == "goodbye" or ints[0]["intent"] == "goodbye_mr" or ints[0]["intent"] == "goodbye_hi"):
            closeLabel = tk.Label(second_frame, text = res + "...", font=('Helvatical bold', 13))
            closeLabel.grid(column=1, row=row_num+1)
            root.after(3000, lambda: root.destroy()) 
        else:
            botLabel = tk.Label(second_frame, text = "Sakecu > ")
            botLabel.grid(column=0, row=row_num)
            lbl.config(text = res)
            self.get_user_res(row_num+1)

        
        



        # vbar.config(command=second_frame.yview)

        # second_frame.config(scrollregion=(0,0,5000,5000))
        # second_frame.pack(side=LEFT, expand=True, fill=BOTH)

        # second_frame.mainloop()



    def get_user_res(self, row_num):
        # while True:

        # second_frame.config(scrollregion=second_frame.bbox("all"))



        #user label
        userLabel = tk.Label(second_frame, text = "You > ")
            
        # TextBox Creation
        inputtxt = tk.Text(second_frame,
                            height = 1,
                            width = 60)

        userLabel.grid(column=0, row=row_num)
        inputtxt.grid(column=1, row=row_num, pady=10)  

        # user_res_list.append(inputtxt.get(1.0, "end-1c"))

        # user_ip = user_res_list[-1]

        # print("Current msg: ", inputtxt.get(1.0, "end-1c"))
            
        # inputtxt.pack()
            
        # Button Creation
        printButton = tk.Button(second_frame,
                                    text = "SEND", 
                                    command=partial(self.get_bot_res, inputtxt, row_num+1))

        printButton.grid(column=2, row=row_num)

        # second_frame.mainloop()



            # printButton.pack()
            
            # Label Creation
        


            # print('Sakecu > ', res)
            # print()
        # i+=1


print()
print()
print("***** SAKEC Bot is Running! *****")
user_res_list = []
chatapp = ChatApp()
chatapp.get_user_res(4)
root.mainloop()