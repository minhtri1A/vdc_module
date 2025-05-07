import whisper
import os
import time

WHISPER_MODEL = None


def load_model_speech_to_text(model="turbo"):
    global WHISPER_MODEL
    start = time.time()
    print("*****Start loading model whisper:", model)
    WHISPER_MODEL = whisper.load_model(model)
    print("*****Load model whisper success:", WHISPER_MODEL, "time", time.time() - start)


def generate_speech_to_text(stt_audio: str):
    if WHISPER_MODEL is None:
        print("*****You need to load the model before doing the next step!!! - call function load_model_speech_to_text")
        return "You need to load the model before doing the next step!!! - call function load_model_speech_to_text"
    # # load audio and pad/trim it to fit 30 seconds
    # audio = whisper.load_audio(audio_file)
    # audio = whisper.pad_or_trim(audio)

    # # make log-Mel spectrogram and move to the same device as the model
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # # detect the spoken language
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # # decode the audio
    # options = whisper.DecodingOptions(language="vi")
    # result = whisper.decode(model, mel, options)

    # # print the recognized text
    # print(result.text)
    start = time.time()
    print("*****Transcribe:", WHISPER_MODEL.device)
    fp16 = WHISPER_MODEL.device == "cuda"
    result = WHISPER_MODEL.transcribe(stt_audio, language="vi", fp16=fp16)
    print("*****Finish:", WHISPER_MODEL.device, "time", time.time() - start)
    print(result["text"])
    return result["text"]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAKER_DIR = os.path.join(SCRIPT_DIR, "speakers")

if __name__ == "__main__":
    file_path = os.path.join(SPEAKER_DIR, "vi_sample.wav")
    load_model_speech_to_text()
    generate_speech_to_text(file_path)
