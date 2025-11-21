# Video Subtitle Generator

This project provides a tool for automatically generating subtitles for videos, supporting both transcription and translation into multiple languages using **Whisper** and **Helsinki-NLP MarianMT** models.

## Features

- Extracts audio from video files.
- Detects the language of the audio and transcribes it into subtitles.
- Optionally translates the subtitles to another language (English <-> French).
- Generates SRT subtitle files.

## Requirements

- Python 3.11+
- `whisper` (OpenAI's Whisper model)
- `torch` (PyTorch)
- `moviepy` (For video and audio processing)
- `transformers` (For translation models)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ekomlenovic/AI-W-Subtitles.git
   cd AI-W-Subtitles
   ```

2. Install the required packages with pip or conda:

   ```bash
    pip install -r requirements.txt
    ```
    or
    ```bash
    conda env create -f environment.yml
    conda activate subtitles
    ```

## Usage

The script can be run from the command line using the following syntax:

```bash
python main.py [--source_lang <language>] [--translate <fr/en/ja/zh>] ./path/to/video.<mp4/mkv/avi>
```

### Arguments:

- `--model_name`: (Optional) The Whisper model to use. Default is turbo. Available options: `tiny`, `base`, `small`, `medium`, `large`, `turbo`.
[Choose the model.](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages)
- `--source_lang`: (Optional) Manually specify the source language of the video (e.g., en, zh, ja, fr, de, es, etc.). If not specified, the language will be auto-detected by Whisper.
- `--translate`: (Optional) The language to translate the subtitles to. Options: `fr`, `en`, `ja`, `zh` (French, English, Japanese, Chinese).

### Examples:

```bash
# Auto-detect language and generate subtitles only
python main.py video.mp4

# Specify source language as Chinese and generate subtitles
python main.py --source_lang zh video.mp4

# Auto-detect language and translate to English
python main.py --translate en video.mp4

# Specify source as Japanese and translate to Chinese
python main.py --source_lang ja --translate zh video.mp4

# Use large model with manual language specification
python main.py --model_name large --source_lang fr --translate en video.mp4
```

### Output
The generated subtitle files will be saved in a folder named `subtitles`, containing:

- Original subtitles in the detected language (e.g., `video_fr.srt`).
- Translated subtitles in the specified language (e.g., `video_en.srt`), if translation is requested.


### Documentation website generated with MkDocs in site folder
Building the documentation website.
```bash
mkdocs build
```
The documentation website is generated with MkDocs and is available in the `site` folder. To view the documentation, open the `index.html` file in your browser.

## Contributing

Feel free to fork the repository, open issues, and submit pull requests. Contributions are always welcome!