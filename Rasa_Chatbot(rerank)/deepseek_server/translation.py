# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load model GemmaX2-28-2B-v0.1
# model_name = "ModelSpace/GemmaX2-28-2B-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# def translate(text, src_lang="vi", tgt_lang="en"):
#     """ Dịch văn bản giữa tiếng Anh và tiếng Việt """
#     lang_token = "<vi2en>" if src_lang == "vi" else "<en2vi>"
#     inputs = tokenizer(lang_token + text, return_tensors="pt", padding=True, truncation=True)
#     translated_tokens = model.generate(**inputs, max_length=512)
#     return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]