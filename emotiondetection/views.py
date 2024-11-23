from django.http import JsonResponse
import json
import joblib
from django.views.decorators.csrf import csrf_exempt
import whisper
import os
from tempfile import NamedTemporaryFile

# Load the pre-trained model
pipe_lr = joblib.load(open("emotiondetection/models/text_emotion.pkl", "rb"))
model = whisper.load_model("base")

@csrf_exempt
def transcribe_audio(request):
    if request.method == "POST":
        # Check if a file is provided
        if 'file' not in request.FILES:
            return JsonResponse({"error": "No file provided"}, status=400)

        # Get the uploaded file
        audio_file = request.FILES['file']

        # Save the file temporarily
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            for chunk in audio_file.chunks():
                temp_audio_file.write(chunk)
            temp_audio_path = temp_audio_file.name

        try:
            # Load and process the audio using Whisper
            audio = whisper.load_audio(temp_audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # Detect language
            _, probs = model.detect_language(mel)
            detected_language = max(probs, key=probs.get)

            # Decode transcription
            options = whisper.DecodingOptions()
            result = whisper.decode(model, mel, options)

            # Return the transcription
            response_data = {
                "detected_language": detected_language,
                "transcription": result.text
            }
            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    return JsonResponse({"error": "Invalid request method"}, status=405)

@csrf_exempt
def detect_emotion(request):
    if request.method == "POST":
        try:
            # Parse the JSON body
            data = json.loads(request.body)
            text = data.get("text", "")

            if not text:
                return JsonResponse({"error": "Text is required"}, status=400)

            # Predict the emotion
            results = pipe_lr.predict([text])
            emotion = results[0]

            if emotion == "joy":
                animation = "TalkingThree"
            elif emotion == "sadness":
                animation = "SadIdle"
            elif emotion == "anger":
                animation = "Angry"
            elif emotion =="fear":
                animation = "Defeated"
            elif emotion=="surprise":
                animation = "Surprised"
            elif emotion == "disgust":
                animation = "DismissingGesture"
            elif emotion == "neutral":
                animation = "TalkingThree"
            elif emotion == "shame":
                animation = "DismissingGesture"

            # Return the detected emotion
            return JsonResponse({"facialExpression": emotion,"animation":animation}, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
