import whisper

def whisper_text():
    model = whisper.load_model("base")
    result = model.transcribe(r"D:\Research\DL Depression\EmoRec\backend\video.webm")
    if result["text"]:
        return result["text"]
    else:
        return "null"