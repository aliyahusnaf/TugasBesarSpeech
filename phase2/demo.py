import argparse
import sys
from pathlib import Path

from phase2.src.pipeline import FINAL_CONFIG, PRETRAINED_CONFIG, TranscriptionPipeline

MODEL_CHOICES = {
    "Whisper small fine-tuned (final)": FINAL_CONFIG,
    "Whisper small pretrained": PRETRAINED_CONFIG,
}


def run_cli(audio_paths, config_path, normalized):
    pipeline = TranscriptionPipeline(config_path)
    failed = 0
    for path in audio_paths:
        try:
            result = pipeline.transcribe(path)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            print(f"Gagal: {error}", file=sys.stderr)
            failed += 1
            continue
        text = result.normalized_text if normalized else result.text
        print(f"{path} ({result.duration_sec:.1f} detik audio, {result.elapsed_sec:.1f} detik proses)")
        print(text)
        print()
    return 1 if failed else 0


def run_ui(port, share):
    import gradio as gr

    loaded = {}

    def get_pipeline(model_name):
        if model_name not in loaded:
            loaded[model_name] = TranscriptionPipeline(MODEL_CHOICES[model_name])
        return loaded[model_name]

    def transcribe(audio_path, model_name):
        if not audio_path:
            raise gr.Error("Upload atau rekam audio dulu.")
        try:
            result = get_pipeline(model_name).transcribe(audio_path)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            raise gr.Error(str(error))
        info = f"Durasi audio {result.duration_sec:.1f} detik, waktu proses {result.elapsed_sec:.1f} detik"
        return result.text, result.normalized_text, info

    with gr.Blocks(title="ASR Bahasa Palembang") as app:
        gr.Markdown("# ASR Bahasa Palembang\nUpload file audio atau rekam lewat mikrofon, lalu klik Transkripsi.")
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
                model_input = gr.Dropdown(list(MODEL_CHOICES), value=list(MODEL_CHOICES)[0], label="Model")
                button = gr.Button("Transkripsi", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Transkripsi", lines=4)
                normalized_output = gr.Textbox(label="Transkripsi ternormalisasi", lines=4)
                info_output = gr.Markdown()
        button.click(transcribe, [audio_input, model_input], [text_output, normalized_output, info_output])

    app.launch(server_port=port, share=share)


def main():
    parser = argparse.ArgumentParser(description="Demo transkripsi audio bahasa Palembang")
    parser.add_argument("audio", nargs="*", type=Path, help="file audio (wav, mp3, m4a, ...)")
    parser.add_argument("--config", default=str(FINAL_CONFIG), help="config model untuk mode CLI")
    parser.add_argument("--normalized", action="store_true", help="tampilkan teks yang sudah dinormalisasi")
    parser.add_argument("--ui", action="store_true", help="jalankan UI web Gradio")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="buat link publik Gradio")
    args = parser.parse_args()

    if args.ui:
        run_ui(args.port, args.share)
    elif args.audio:
        sys.exit(run_cli(args.audio, args.config, args.normalized))
    else:
        parser.error("berikan file audio atau pakai --ui")


if __name__ == "__main__":
    main()
