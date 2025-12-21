import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

texts = [
    "هذا المنتج ممتاز جدًا والجودة رائعة",             # positive
    "الخدمة عادية وليست سيئة",                       # neutral
    "لم يعجبني المنتج على الإطلاق والخدمة سيئة",      # negative
    "سعيد جدًا بالشراء والتجربة كانت رائعة",          # positive
    "المكان نظيف لكن الأسعار مرتفعة قليلاً",         # neutral
    "الخدمة سيئة جدًا ولن أعود مرة أخرى",            # negative
    "التطبيق يعمل بسرعة وسهل الاستخدام",            # positive
    "المنتج لا بأس به لكنه ليس رائعًا",              # neutral
    "الطعام فاسد والخدمة سيئة",                     # negative
    "التجربة كانت ممتعة جدًا والشكر للفريق",         # positive
    "الجودة مقبولة ولكن يمكن تحسينها",              # neutral
    "لم أكن راضيًا عن الخدمة وسأتقدم بشكوى",        # negative
    "المنتج ممتاز وسأوصي به لأصدقائي",              # positive
    "المكان جيد لكن الانتظار طويل",                  # neutral
    "الخدمة أسوأ مما توقعت",                         # negative
    "تجربة رائعة وأعجبني كل شيء",                   # positive
    "الجودة متوسطة والأسعار مرتفعة",                # neutral
    "الموظفون غير متعاونين والخدمة سيئة",          # negative
    "منتج رائع جدًا وفعال",                          # positive
    "المنتج مقبول ولم أشعر بالرضا التام"             # neutral
]


# lowercase labels to match model's config
label_map = {
    "negative": -1,
    "neutral": 0,
    "positive": 1
}

for text in texts:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(torch.softmax(outputs.logits, dim=1), dim=1).item()
    label_name = model.config.id2label[pred_id].lower()  # convert to lowercase
    final_label = label_map[label_name]
    
    print(f"Text: {text}")
    print(f"Predicted sentiment: {label_name} → {final_label}")
    print("-" * 50)
