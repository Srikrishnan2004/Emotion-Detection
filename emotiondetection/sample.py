import whisper

model = whisper.load_model("base")
audio=whisper.load_audio("Recording.mp3")
audio=whisper.pad_or_trim(audio)

mel=whisper.log_mel_spectrogram(audio).to(model.device)

_,probs=model.detect_language(mel)
print(f"Detected Language: {max(probs,key=probs.get)}")

options=whisper.DecodingOptions()
result=whisper.decode(model,mel,options)
print(result.text)


