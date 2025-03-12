# import re
# from deep_translator import GoogleTranslator

# def load_no_translate_words(file_path="no_translate_words.txt"):
#     """Đọc danh sách từ không dịch từ file."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         words = [line.strip() for line in f if line.strip()]
#     return words

# def protect_no_translate(text, no_translate_words):
#     """Thay thế các từ không dịch bằng placeholder trước khi dịch."""
#     placeholders = {}
#     for i, word in enumerate(no_translate_words):
#         pattern = r'\b' + re.escape(word) + r'\b'  # Bảo vệ từ nguyên vẹn
#         placeholder = f"__PROTECT_{i}__"
#         text = re.sub(pattern, placeholder, text)
#         placeholders[placeholder] = word
#     return text, placeholders

# def restore_no_translate(translated_text, placeholders):
#     """Khôi phục các từ không dịch từ placeholder."""
#     for placeholder, word in placeholders.items():
#         translated_text = translated_text.replace(placeholder, word)
#     return translated_text

# def translate_en_to_vi(text, no_translate_words):
#     """Dịch từ tiếng Anh sang tiếng Việt nhưng giữ nguyên các từ không dịch."""
#     text, placeholders = protect_no_translate(text, no_translate_words)  # Bảo vệ từ không dịch
#     translated_text = GoogleTranslator(source='en', target='vi').translate(text)  # Dịch
#     translated_text = restore_no_translate(translated_text, placeholders)  # Khôi phục từ không dịch
#     return translated_text