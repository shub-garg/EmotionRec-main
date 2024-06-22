from transformers import AutoTokenizer, BertForMultipleChoice
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import transformers

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForMultipleChoice.from_pretrained("bert-base-uncased")

# def text_classify(prompt):
#     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#     encoding = tokenizer([prompt for i in range(len(emotion_labels))], emotion_labels, return_tensors="pt", padding=True)
#     labels = torch.tensor(0).unsqueeze(0)
#     outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  # batch size is 1

#     # the linear classifier still needs to be trained
#     logits = outputs.logits
#     prediction = torch.nn.functional.softmax(logits, dim=1)

#     labels = {}
#     for i in range(len(emotion_labels)):
#         labels[emotion_labels[i]] = round(prediction[0][i].item(),3)
    
#     return labels

def text_classify(prompt):

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    tokenizer = AutoTokenizer.from_pretrained("lordtt13/emo-mobilebert")
    model = AutoModelForSequenceClassification.from_pretrained("lordtt13/emo-mobilebert")

    nlp_sentence_classif = transformers.pipeline('sentiment-analysis', model = model, tokenizer = tokenizer)
    return nlp_sentence_classif(prompt)
# Output: [{'label': 'sad', 'score': 0.93153977394104}]


