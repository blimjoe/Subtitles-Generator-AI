import whisper
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
from datetime import timedelta
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import platform

class SubtitleGenerator:
    def __init__(self, model_name: str = "turbo", use_gpu: bool = True):
        """Initialize the SubtitleGenerator with Whisper model and lazy-loaded translation models.

        Args:
            model_name (str): The model size to use for Whisper. Default is "turbo".
            use_gpu (bool): Whether to use GPU acceleration. Default is True.
        """
        # Setup device for optimal Apple M4 performance
        self.device = self._setup_device(use_gpu)
        
        # Load Whisper model (now fully supports MPS!)
        self.whisper_model = whisper.load_model(model_name, device=self.device)
        
        # Initialize translation model cache - models are loaded lazily when needed
        self._translation_models = {}
        self._translation_tokenizers = {}
        
        # Model configurations for supported language pairs
        self._model_configs = {
            ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
            ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
            ('ja', 'zh'): 'shun89/opus-mt-ja-zh'
        }
        
        print(f"✅ Whisper model '{model_name}' loaded successfully on {self.device}")
        print(f"✅ Translation models will be loaded on-demand for optimal performance")
        print(f"🚀 Full GPU acceleration enabled for Apple M4!")

    def _setup_device(self, use_gpu: bool = True):
        """Setup the optimal device for Apple M4 GPU acceleration."""
        if not use_gpu:
            print("🔧 GPU disabled by user, using CPU")
            return "cpu"
        
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("🚀 Apple Silicon GPU (MPS) detected - enabling GPU acceleration")
            print(f"   Platform: {platform.machine()}")
            try:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except:
                pass
            return "mps"
        elif torch.cuda.is_available():
            print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            print("⚠️  No GPU acceleration available, using CPU")
            return "cpu"

    def _load_translation_model(self, source_lang: str, target_lang: str):
        """Lazy load translation model and tokenizer."""
        lang_pair = (source_lang, target_lang)
        
        if lang_pair in self._translation_models:
            return self._translation_models[lang_pair], self._translation_tokenizers[lang_pair]
        
        if lang_pair not in self._model_configs:
            return None, None
        
        model_name = self._model_configs[lang_pair]
        
        try:
            print(f"Loading translation model for {source_lang} -> {target_lang}...")
            model = MarianMTModel.from_pretrained(model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            
            if self.device != "cpu":
                try:
                    model = model.to(self.device)
                    print(f"✓ Model moved to {self.device}")
                except:
                    model = model.to("cpu")
            
            self._translation_models[lang_pair] = model
            self._translation_tokenizers[lang_pair] = tokenizer
            
            print(f"✓ Translation model {source_lang} -> {target_lang} cached")
            return model, tokenizer
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            return None, None

    def get_cache_info(self) -> dict:
        """Get cached translation models info."""
        return {
            "loaded_models": list(self._translation_models.keys()),
            "supported_pairs": list(self._model_configs.keys()),
            "cache_size": len(self._translation_models)
        }

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video."""
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)
            video.close()
            return True
        except Exception as e:
            print(f"Error during audio extraction: {str(e)}")
            return False

    def detect_language(self, text: str) -> str:
        """Detect language using Whisper."""
        audio_features = self.whisper_model.embed_audio(text)
        return self.whisper_model.detect_language(audio_features)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between languages."""
        try:
            model, tokenizer = self._load_translation_model(source_lang, target_lang)
            
            if model is None or tokenizer is None:
                raise ValueError(f"Translation not supported: {source_lang} to {target_lang}")

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            if hasattr(model, 'device') and model.device.type != 'cpu':
                inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            translated = model.generate(**inputs)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            return text

    def generate_subtitles(self, audio_path: str) -> tuple:
        """Generate subtitles from audio using Whisper."""
        try:
            result = self.whisper_model.transcribe(audio_path, verbose=False)
            source_lang = result['language']
            print(f"Detected language: {source_lang}")
            return result, source_lang
        except Exception as e:
            print(f"Error during subtitle generation: {str(e)}")
            return None, None

    @staticmethod
    def format_time(seconds: float) -> str:
        """Convert seconds to SRT format."""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds // 60) % 60
        secs = td.seconds % 60
        milliseconds = round(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def create_srt(self, transcription: dict, output_path: str, target_lang: str = None) -> bool:
        """Create SRT file with optional translation."""
        try:
            source_lang = transcription['language']
            with open(output_path, 'w', encoding='utf-8') as f:
                segments = transcription['segments']
                for i, segment in tqdm(enumerate(segments, 1), desc="Creating SRT file"):
                    start_time = self.format_time(segment['start'])
                    end_time = self.format_time(segment['end'])
                    text = segment['text'].strip()

                    if target_lang and source_lang != target_lang:
                        translated_text = self.translate_text(text, source_lang, target_lang)
                        text = translated_text

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            return True
        except Exception as e:
            print(f"Error during SRT file creation: {str(e)}")
            return False

    def process_video(self, video_path: str, target_lang: str = None) -> None:
        """Main subtitle generation process."""
        output_dir = "subtitles"
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, "temp_audio.wav")
        
        print("Starting subtitle generation process...")
        
        print("Extracting audio...")
        if not self.extract_audio(video_path, audio_path):
            return
        
        print("Generating subtitles...")
        transcription, source_lang = self.generate_subtitles(audio_path)
        if transcription is None:
            return
 
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        original_srt = os.path.join(output_dir, f"{base_name}_{source_lang}.srt")
        self.create_srt(transcription, original_srt)
        print(f"Original subtitles generated: {original_srt}")
        
        if target_lang:
            translated_srt = os.path.join(output_dir, f"{base_name}_{target_lang}.srt")
            self.create_srt(transcription, translated_srt, target_lang)
            print(f"Translated subtitles generated: {translated_srt}")
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        print("Process completed!")
