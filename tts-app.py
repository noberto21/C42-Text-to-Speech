# Importing Required Libraries
import gradio as gr
from TTS.api import TTS
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import librosa.display
import warnings
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Suppress specific warnings
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token.")

# Loading the Coqui TTS Model
model_name = TTS.list_models()[0] #using the first available model
tts = TTS(model_name)

# Voice Selection Options
available_speakers = [
    "Daisy Studious", "Sofia Hellen", "Asya Anara",
    "Eugenio Mataracı", "Viktor Menelaos", "Damien Black"
]

language_mapping = {
    "US English": "en",
    "Spanish (LatAm)": "es",
    "German": "de",
    "French": "fr",
    "Japanese": "ja",
    "Hindi": "hi",
    "Mandarin": "zh-cn",
    "Italian": "it"
}

available_languages = list(language_mapping.keys())

# Create output directory
os.makedirs("output", exist_ok=True)

# Global variables
last_generated_audio = None
last_generated_text = ""
last_speaker = ""
last_language = ""

# Custom CSS for animations
custom_css = """
/* Pulse animation for generate button */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* Fade-in animation */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Wave animation for audio player */
.wave-animation {
    position: relative;
    height: 50px;
    display: flex;
    align-items: center;
}

.wave-bar {
    width: 4px;
    height: 20px;
    margin: 0 2px;
    background: linear-gradient(to top, #1e88e5, #0d47a1);
    border-radius: 2px;
    animation: wave 1.2s ease-in-out infinite;
}

@keyframes wave {
    0%, 100% { height: 20px; }
    50% { height: 40px; }
}

/* Apply animations */
.generate-btn {
    animation: pulse 2s infinite;
}

.fade-in {
    animation: fadeIn 0.8s ease-out;
}

/* Glowing border effect */
.glow-border {
    box-shadow: 0 0 10px rgba(30, 136, 229, 0.5);
    transition: box-shadow 0.3s ease;
}

.glow-border:hover {
    box-shadow: 0 0 15px rgba(30, 136, 229, 0.8);
}

/* Animated tabs */
.tab-button {
    transition: all 0.3s ease;
}

.tab-button:hover {
    transform: translateY(-3px);
}

/* Audio visualization animation */
.audio-viz-container {
    transition: all 0.5s ease;
}

.audio-viz-container:hover {
    transform: scale(1.01);
}
"""

def trim_text(text: str, max_length: int = 200) -> str:
    return text[:max_length] + "..." if len(text) > max_length else text

def generate_speech_with_timestamps(text: str, speaker: str, language: str):
    global last_generated_audio, last_generated_text, last_speaker, last_language
    
    timestamp = int(time.time())
    output_path = f"output/generated_speech_{timestamp}.wav"
    
    start_time = time.time()
    tts_language = language_mapping.get(language, "en")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            tts.tts_to_file(
                text=text,
                speaker=speaker.split("(")[0].strip(),
                language=tts_language,
                file_path=output_path
            )
            
    except Exception as e:
        raise gr.Error(f"Speech generation failed: {str(e)}")
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    
    last_generated_audio = output_path
    last_generated_text = text
    last_speaker = speaker
    last_language = language
    
    samplerate, data = wavfile.read(output_path)
    speech_length = len(data) / samplerate
    
    return output_path, len(text.split()), speaker, language, round(speech_length, 2), duration

def generate_visualizations():
    global last_generated_audio
    
    if not last_generated_audio or not os.path.exists(last_generated_audio):
        return None, None, "No audio file found"
    
    try:
        # Waveform
        samplerate, data = wavfile.read(last_generated_audio)
        duration = len(data) / samplerate
        time_axis = np.linspace(0, duration, num=len(data))
        
        fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#1E1E1E')
        ax1.plot(time_axis, data, color='#1f77b4', alpha=0.8, linewidth=1.2)
        
        ax1.set_facecolor('#2E2E2E')
        ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        for spine in ax1.spines.values():
            spine.set_color('white')
        ax1.tick_params(colors='white')
        ax1.set_xlabel("Time (seconds)", color='white')
        ax1.set_ylabel("Amplitude", color='white')
        ax1.set_title("Waveform", color='white', fontsize=12, pad=10)
        
        waveform_path = f"output/waveform_{int(time.time())}.png"
        plt.savefig(waveform_path, bbox_inches='tight', transparent=True, dpi=120)
        plt.close(fig1)
        
        # Spectrogram
        y, sr = librosa.load(last_generated_audio)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor='#1E1E1E')
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2)
        
        ax2.set_facecolor('#2E2E2E')
        for spine in ax2.spines.values():
            spine.set_color('white')
        ax2.tick_params(colors='white')
        ax2.set_xlabel("Time (seconds)", color='white')
        ax2.set_ylabel("Frequency (Hz)", color='white')
        ax2.set_title("Spectrogram", color='white', fontsize=12, pad=10)
        
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        
        spectrogram_path = f"output/spectrogram_{int(time.time())}.png"
        plt.savefig(spectrogram_path, bbox_inches='tight', transparent=True, dpi=120)
        plt.close(fig2)
        
        # Audio features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        metadata = (
            f"📊 Audio Analysis:\n"
            f"⏱️ Duration: {duration:.2f} seconds\n"
            f"• Sample Rate: {sr} Hz\n"
            f"• Spectral Centroid: {np.mean(spectral_centroid):.2f} Hz (avg)\n"
            f"• Spectral Bandwidth: {np.mean(spectral_bandwidth):.2f} Hz (avg)\n\n"
            f"🗣️ Speaker: {last_speaker.split('(')[0].strip()}\n"
            f"🌍 Language: {last_language}\n"
            f"📝 Text: '{trim_text(last_generated_text)}'"
        )
        
        return waveform_path, spectrogram_path, metadata
    
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def generate_speech(text: str, speaker: str, language: str):
    if not text.strip():
        return None, "Please enter text", "", gr.update(interactive=False), gr.update(visible=False)
    
    try:
        audio_path, word_count, speaker_name, lang, speech_length, duration = generate_speech_with_timestamps(text, speaker, language)
        
        info_text = (
            f"📝 Word Count: {word_count}\n"
            f"🗣️ Voice: {last_speaker.split('(')[0].strip()}\n"
            f"🌍 Language: {lang}\n"
            f"⏱️ Audio Length: {speech_length} seconds\n"
            f"⚡ Generation Time: {duration} seconds"
        )
        
        return audio_path, info_text, "✅ Success!", gr.update(interactive=True), gr.update(visible=True)
    
    except Exception as e:
        return None, str(e), "❌ Error", gr.update(interactive=False), gr.update(visible=False)

def setup_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
        title="Animated TTS Generator",
        css=custom_css
    ) as app:
        
        # Header with animation
        with gr.Column(elem_classes="fade-in"):
            gr.Markdown("""
            <div style="text-align: center;">
                <h1 style="color: #1e88e5; font-weight: 800; margin-bottom: 0.5rem;">🎤 Text-to-Speech GenAI with Coqui TTS</h1>
                <p style="color: #666; font-size: 1.1rem;">Convert text to speech using Coqui TTS with support for different languages and speakers.</p>
            </div>
            """)
        
        with gr.Row():
            # Input Column
            with gr.Column(scale=1, elem_classes="fade-in"):
                with gr.Group(elem_classes="glow-border"):
                    text_input = gr.Textbox(
                        label="Enter Text",
                        placeholder="Type or paste your text here...",
                        lines=5,
                        max_length=500,
                        elem_id="text_input"
                    )
                    
                    with gr.Row():
                        speaker_dropdown = gr.Dropdown(
                            choices=available_speakers,
                            value=available_speakers[0],
                            label="Select Speaker",
                            interactive=True
                        )
                        language_radio = gr.Radio(
                            choices=available_languages,
                            value=available_languages[0],
                            label="Select Localization",
                            interactive=True
                        )
                    
                    generate_btn = gr.Button(
                        "Generate Speech",
                        variant="primary",
                        elem_classes="generate-btn"
                    )
                
                with gr.Accordion("Generation Details", open=False):
                    data_info = gr.Textbox(
                        label="Analysis Results",
                        interactive=False,
                        lines=8
                    )
                
                status_message = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False
                )
            
            # Output Column
            with gr.Column(scale=1, elem_classes="fade-in"):
                with gr.Group(elem_classes="glow-border"):
                    audio_output = gr.Audio(
                        label="Generated Speech",
                        interactive=False,
                        elem_classes="audio-viz-container"
                    )
                
                with gr.Tab("Audio Analysis"):
                    gr.Markdown("### Audio Visualizations", elem_classes="fade-in")
                    visualization_btn = gr.Button(
                        "Generate Visualizations",
                        interactive=False,
                        visible=True,
                        variant="secondary",
                        elem_classes="tab-button"
                    )
                    
                    metadata_display = gr.Textbox(
                        label="Audio Characteristics",
                        interactive=False,
                        visible=True,
                        lines=8
                    )
                    
                    with gr.Tabs():
                        with gr.Tab("Waveform", elem_classes="tab-button"):
                            waveform_output = gr.Image(
                                label="Waveform",
                                interactive=False,
                                elem_classes="audio-viz-container"
                            )
                        
                        with gr.Tab("Spectrogram", elem_classes="tab-button"):
                            spectrogram_output = gr.Image(
                                label="Spectrogram",
                                interactive=False,
                                elem_classes="audio-viz-container"
                            )
        
        # Event handlers
        generate_btn.click(
            generate_speech,
            inputs=[text_input, speaker_dropdown, language_radio],
            outputs=[audio_output, data_info, status_message, visualization_btn, metadata_display]
        )
        
        visualization_btn.click(
            generate_visualizations,
            outputs=[waveform_output, spectrogram_output, metadata_display]
        )
        
        speaker_dropdown.change(
            lambda x: gr.update(value=x.split("(")[-1].split(")")[0].strip() if "(" in x and ")" in x else "English"),
            inputs=speaker_dropdown,
            outputs=language_radio
        )
    
    return app

if __name__ == "__main__":
    app = setup_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        favicon_path="icon.png" if os.path.exists("icon.png") else None
    )