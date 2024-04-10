import warnings
warnings.filterwarnings('ignore') 

from transformers import pipeline

# English to Chinese
en_zh_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")

res = en_zh_translator("How old are you?", src_lang='en', tgt_lang='zh') 
print(res)
print(res[0]['translation_text'])

res = en_zh_translator(["Today is Friday.", "what\'s the most important thing?"], src_lang='en', tgt_lang='zh') 
print(res)

# Chinese to English
zh_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

res = zh_en_translator("哎呦，你干嘛？", src_lang='zh', tgt_lang='en') 
print(res[0]['translation_text'])

# Test use default model
default_translator = pipeline("translation_en_to_fr")
res = default_translator("How old are you?") 
print(res[0]['translation_text'])
