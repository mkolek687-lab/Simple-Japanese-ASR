import os
import sys
import whisper
import ffmpeg
import tkinter as tk
from tkinter import filedialog
from typing import Optional

# --- CONFIGURATION ---
# -----------------------------------------------------------------------------
OUTPUT_DIR = r"Z:\ASG\Temp"
MODEL_SIZE = "large-v3"
LANGUAGE = "ja"


# -----------------------------------------------------------------------------


def format_srt_timestamp(seconds: float) -> str:
    """Converts a float number of seconds to the SRT time format HH:MM:SS,ms."""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def select_files() -> list[str]:
    """Open a file dialog for selecting multiple video files."""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    filetypes = [("Video files", "*.mkv *.mp4 *.avi *.mov"), ("All files", "*.*")]
    files = filedialog.askopenfilenames(title="Select video files", filetypes=filetypes)
    return list(files)


def select_audio_stream(video_path: str) -> Optional[int]:
    """Probes a video file for audio streams and prompts the user to select one."""
    print(f"\n🎵 Probing audio streams for: {os.path.basename(video_path)}")
    try:
        probe = ffmpeg.probe(video_path)
        audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']

        if not audio_streams:
            print("❌ No audio streams found in this file.")
            return None

        if len(audio_streams) == 1:
            print("✅ Found 1 audio stream, auto-selecting.")
            return 0

        print("Please select an audio stream:")
        for i, stream in enumerate(audio_streams):
            tags = stream.get('tags', {})
            lang = tags.get('language', 'unknown')
            title = tags.get('title', 'No Title')
            codec = stream.get('codec_name', 'N/A')
            channels = stream.get('channel_layout', 'N/A')
            print(f"  {i + 1}. Language: {lang.upper()}, Title: {title}, Codec: {codec}, Channels: {channels}")

        while True:
            try:
                choice = int(input(f"👉 Enter your choice (1-{len(audio_streams)}): "))
                if 1 <= choice <= len(audio_streams):
                    return choice - 1
                else:
                    print(f"❌ Invalid choice. Please enter a number between 1 and {len(audio_streams)}.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")

    except ffmpeg.Error as e:
        print(f"❌ Error probing file: {e.stderr.decode('utf8')}")
        return None


def extract_audio(video_path: str, audio_output_path: str, stream_index: int) -> bool:
    """Extracts a specific audio stream from a video file and saves it as a WAV file."""
    print(f"🎬 Extracting audio from stream #{stream_index}...")
    try:
        process = (
            ffmpeg
            .input(video_path)
            .output(audio_output_path, map=f'0:a:{stream_index}', format="wav", acodec="pcm_s16le", ac=1, ar='16000', vn=None)
            .global_args("-hide_banner", "-loglevel", "error")
            .run_async(pipe_stderr=True, overwrite_output=True)
        )

        while True:
            line = process.stderr.readline()
            if not line: break
            text = line.decode("utf-8", errors="ignore").strip()
            if "time=" in text and "speed=" in text:
                sys.stdout.write(f"\r   {text}")
                sys.stdout.flush()

        process.wait()
        sys.stdout.write("\n")

        if process.returncode == 0:
            print(f"✅ Audio saved to: {audio_output_path}")
            return True
        else:
            print(f"❌ FFmpeg failed with return code {process.returncode}.")
            return False

    except Exception as e:
        print(f"❌ An error occurred during audio extraction: {e}")
        return False


def transcribe_and_save_srt(model, audio_path: str, output_srt_path: str, language: str):
    """Transcribes an audio file and saves the result as an SRT subtitle file."""
    try:
        print(f"🎤 Transcribing with GPU... (This may take a while)")
        result = model.transcribe(audio_path, language=language)

        with open(output_srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], 1):
                f.write(f"{i}\n")
                start_time = format_srt_timestamp(segment["start"])
                end_time = format_srt_timestamp(segment["end"])
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")

        print(f"📝 Subtitles saved to: {output_srt_path}")

    except Exception as e:
        print(f"❌ An error occurred during transcription: {e}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected_videos = select_files()
    if not selected_videos:
        print("No files selected. Exiting.")
        return

    print(f"\n🧠 Loading Whisper model '{MODEL_SIZE}'... (This may take a moment)")
    try:
        model = whisper.load_model(MODEL_SIZE)
        print("✅ Model loaded successfully. Using GPU." if model.device.type == 'cuda' else "✅ Model loaded successfully. Using CPU.")
    except Exception as e:
        print(f"❌ Failed to load Whisper model: {e}")
        return

    for video_path in selected_videos:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_audio_path = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
        output_srt_path = os.path.join(OUTPUT_DIR, f"{base_name}.srt")

        audio_stream_index = select_audio_stream(video_path)
        if audio_stream_index is None:
            print(f"Skipping file {video_path} due to audio stream issue.")
            continue

        if extract_audio(video_path, temp_audio_path, audio_stream_index):
            transcribe_and_save_srt(model, temp_audio_path, output_srt_path, LANGUAGE)

            try:
                os.remove(temp_audio_path)
                print(f"🗑️ Deleted temporary audio file: {temp_audio_path}")
            except OSError as e:
                print(f"⚠️ Could not delete temporary file {temp_audio_path}: {e}")
        print("-" * 60)

    print("\n✨ All tasks completed!")


if __name__ == "__main__":
    main()
