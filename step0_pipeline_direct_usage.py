import warnings
warnings.filterwarnings('ignore') 

from transformers import pipeline

En_Zh = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
Zh_En = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")

res = En_Zh("How old are you?", src_lang='en', tgt_lang='zh') 
print(res[0]['translation_text'])

res = Zh_En("哎呦，你干嘛？", src_lang='zh', tgt_lang='en') 
print(res[0]['translation_text'])