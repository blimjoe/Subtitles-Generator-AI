from SubtitleGenerator import SubtitleGenerator
import argparse


def main():
    """Main function to handle command-line arguments and process the video."""
    parser = argparse.ArgumentParser(description="Video Subtitle Generator with GPU Acceleration")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--model_name", default="turbo", choices=["tiny", "base", "small", "medium", "large", "turbo"], help="Whisper model name")
    parser.add_argument("--translate", choices=["fr", "en", "ja", "zh"], help="Language to translate subtitles to (fr: French, en: English, ja: Japanese, zh: Chinese)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration and use CPU only")

    args = parser.parse_args()

    print(f"Processing video: {args.video_path}")
    print(f"Whisper model: {args.model_name}")
    if args.translate:
        print(f"Translating subtitles to: {args.translate}")
    
    use_gpu = not args.no_gpu
    generator = SubtitleGenerator(args.model_name, use_gpu=use_gpu)
    generator.process_video(args.video_path, args.translate)

if __name__ == "__main__":
    main()
