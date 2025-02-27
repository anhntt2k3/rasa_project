# # translation.py
# from transformers import MarianMTModel, MarianTokenizer

# # Load model dịch Anh-Việt và Việt-Anh
# model_name_en_vi = "Helsinki-NLP/opus-mt-en-vi"
# model_name_vi_en = "Helsinki-NLP/opus-mt-vi-en"

# tokenizer_en_vi = MarianTokenizer.from_pretrained(model_name_en_vi)
# model_en_vi = MarianMTModel.from_pretrained(model_name_en_vi)

# tokenizer_vi_en = MarianTokenizer.from_pretrained(model_name_vi_en)
# model_vi_en = MarianMTModel.from_pretrained(model_name_vi_en)

# def translate(text, src_lang="vi", tgt_lang="en"):
#     """ Dịch văn bản giữa tiếng Anh và tiếng Việt """
#     tokenizer, model = (tokenizer_vi_en, model_vi_en) if src_lang == "vi" else (tokenizer_en_vi, model_en_vi)
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#     translated_tokens = model.generate(**inputs)
#     return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
