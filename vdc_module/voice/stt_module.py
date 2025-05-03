import whisper
import os


def generate_text(stt_audio: str, gpu=False):
    model = whisper.load_model("medium")
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
    result = model.transcribe(stt_audio, language="vi", fp16=gpu)
    print(result["text"])

    return result["text"]


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAKER_DIR = os.path.join(SCRIPT_DIR, "speakers")

if __name__ == "__main__":
    file_path = os.path.join(SPEAKER_DIR, "vi_sample.wav")
    generate_text(file_path)
