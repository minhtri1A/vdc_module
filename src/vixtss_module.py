import os
import torch
import torchaudio
import subprocess
from huggingface_hub import snapshot_download, hf_hub_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from vinorm import TTSnorm
from underthesea import sent_tokenize
import string
from unidecode import unidecode
from datetime import datetime

XTTS_MODEL = None
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FILTER_SUFFIX = "_DeepFilterNet3.wav"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define dictionaries to store cached results
cache_queue = []
speaker_audio_cache = {}
filter_cache = {}
conditioning_latents_cache = {}

#-----helper-----#
def normalize_vietnamese_text(text):
    text = (
       text
        .replace("..", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("AI", "Ây Ai")
        .replace("A.I", "Ây Ai")
        .replace("vuadungcu.com", "vua dụng cụ chấm cơm")
    )
    return text

def calculate_keep_len(text, lang):
    """Simple hack for short sentences"""
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = text.count(".") + text.count("!") + text.count("?") + text.count(",")

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1

def invalidate_cache(cache_limit=50):
    """Invalidate the cache for the oldest key"""
    if len(cache_queue) > cache_limit:
        key_to_remove = cache_queue.pop(0)
        print("Invalidating cache", key_to_remove)
        if os.path.exists(key_to_remove):
            os.remove(key_to_remove)
        if os.path.exists(key_to_remove.replace(".wav", "_DeepFilterNet3.wav")):
            os.remove(key_to_remove.replace(".wav", "_DeepFilterNet3.wav"))
        if key_to_remove in filter_cache:
            del filter_cache[key_to_remove]
        if key_to_remove in conditioning_latents_cache:
            del conditioning_latents_cache[key_to_remove]

def get_file_name(text, max_char=50):
    filename = text[:max_char]
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename.translate(
        str.maketrans("", "", string.punctuation.replace("_", ""))
    )
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    filename = f"{current_datetime}_{filename}"
    return filename

#-----handle-----#
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(checkpoint_dir="src/model/", repo_id="capleaf/viXTTS", use_deepspeed=False):
    global XTTS_MODEL
    clear_gpu_cache()
    os.makedirs(checkpoint_dir, exist_ok=True)

    required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
    files_in_dir = os.listdir(checkpoint_dir)
    if not all(file in files_in_dir for file in required_files):
        print("******>Đang tải model viXTTS từ Hugging Face...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=checkpoint_dir,
        )
        hf_hub_download(
            repo_id="coqui/XTTS-v2",
            filename="speakers_xtts.pth",
            local_dir=checkpoint_dir,
        )
        print("******>Tải xong!")

    xtts_config = os.path.join(checkpoint_dir, "config.json")
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("******>Đang khởi tạo model...")
    XTTS_MODEL.load_checkpoint(
        config, checkpoint_dir=checkpoint_dir, use_deepspeed=use_deepspeed
    )
    XTTS_MODEL.eval()
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("******>Model đã sẵn sàng!")

# lang: language
# tts_text: text to speech, 
# speaker_audio_file: sample voice 
# use_deepfilter:
# normalize_text:
def generate_voice(lang:str, tts_text:str, speaker_audio_file:str, use_deepfilter:bool, normalize_text:bool):
    global filter_cache, conditioning_latents_cache, cache_queue
    
    if XTTS_MODEL is None:
        return "You need to run the previous step to load the model !!", None, None
    
    if not speaker_audio_file:
        return "You need to provide reference audio!!!", None, None
    
    # check set key cache for file
    speaker_audio_key = speaker_audio_file
    if not speaker_audio_key in cache_queue:
        cache_queue.append(speaker_audio_key)
        invalidate_cache()
    
    # check set cache for file and deepfilter(loc tieng on am thanh)
    if use_deepfilter and speaker_audio_key in filter_cache:
        print("******>Using filter cache...")
        speaker_audio_file = filter_cache[speaker_audio_key]
    elif use_deepfilter:
        print("******>Running filter...")
        subprocess.run(
            [
                "deepFilter",
                speaker_audio_file,
                "-o",
                os.path.dirname(speaker_audio_file),
            ]
        )
        filter_cache[speaker_audio_key] = speaker_audio_file.replace(
            ".wav", FILTER_SUFFIX
        )
        speaker_audio_file = filter_cache[speaker_audio_key]
    
    #check set cache for Conditioning latents(lay dac trung cua giong noi)
    cache_key = (
        speaker_audio_key,
        XTTS_MODEL.config.gpt_cond_len,
        XTTS_MODEL.config.max_ref_len,
        XTTS_MODEL.config.sound_norm_refs,
    )

    if cache_key in conditioning_latents_cache:
        print("******>Using conditioning latents cache...")
        gpt_cond_latent, speaker_embedding = conditioning_latents_cache[cache_key]
    else:
        print("******>Computing conditioning latents...")
        gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
            audio_path = speaker_audio_file,
            gpt_cond_len = XTTS_MODEL.config.gpt_cond_len,
            max_ref_length = XTTS_MODEL.config.max_ref_len,
            sound_norm_refs = XTTS_MODEL.config.sound_norm_refs,
        )
        conditioning_latents_cache[cache_key] = (gpt_cond_latent, speaker_embedding)

    # normalize_text vietnamese(chuan hoa tieng viet lai) -> 12kg = 12 kilogram,...
    if normalize_text and lang == "vi":
        tts_text = normalize_vietnamese_text(tts_text)

    # Split text by sentence(chunk)
    if lang in ["ja", "zh-cn"]:
        sentences = tts_text.split("。")
    else:
        sentences = sent_tokenize(tts_text)

    # create wav chunk from sentences
    wav_chunks = []
    for sentence in sentences:
        if sentence.strip() == "":
            continue
        wav_chunk = XTTS_MODEL.inference(
            text=sentence,
            language=lang,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            # The following values are carefully chosen for viXTTS
            temperature=0.3,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=30,
            top_p=0.85,
            enable_text_splitting=True,
        )
        keep_len = calculate_keep_len(sentence, lang)
       
        wav_chunk["wav"] = wav_chunk["wav"][:keep_len]

        #convert wav to tensor
        wav_chunks.append(torch.tensor(wav_chunk["wav"]))
      

    # save file
    out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
    print("*****out_wav", out_wav)
    gr_audio_id = os.path.basename(os.path.dirname(speaker_audio_file))
    out_path = os.path.join(OUTPUT_DIR, f"{get_file_name(tts_text)}_{gr_audio_id}.wav")

    torchaudio.save(out_path, out_wav, 24000)

    print("******>Saving output to ", out_path)


lang_demo = "vi"
text_demo = "Xin chào, tôi là vua dụng cụ AI, được viết bởi vuadungcu.com, Bạn có thể hỏi tôi bất cứ thứ gì trên đời này, trả lời được hay không thì hên xui."
speaker_demo = f"{MODEL_DIR}/samples/nguyenngocngan_sample.wav"

if __name__ == "__main__":
    print("******>speaker demo ", speaker_demo)
    load_model()
    generate_voice(lang_demo, text_demo, speaker_demo, True, True)