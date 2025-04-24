# Import the necessary libraries
from TTS.api import TTS
import os

# Load the TTS model
model_name = TTS.list_models()[0]  # Use the first available model
tts = TTS(model_name)

# Select the speaker and language
selected_speaker = tts.speakers[0]
selected_language = tts.languages[0]

# Set output folder and input text
os.makedirs("output", exist_ok=True)
text = "First, solve the problem. Then, write the code."

# Generate the speech and save it to a WAV file
tts.tts_to_file(
    text=text,
    speaker=selected_speaker,  # Provide speaker name
    language=selected_language,  # Provide language code
    file_path="output/output.wav"
)
print("TTS complete! Check the output folder for the audio file.")

# Import the necessary libraries
from TTS.api import TTS
import os

# Load the TTS model
model_name = TTS.list_models()[0]  # Use the first available model
tts = TTS(model_name)

# Select the speaker and language
selected_speaker = tts.speakers[0]
selected_language = tts.languages[0]

# Set output folder and input text
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
text = "First, solve the problem. Then, write the code."

# Generate the speech and save it to a WAV file
output_file_path = os.path.join(output_folder, "output.wav")
tts.tts_to_file(
    text=text,
    speaker=selected_speaker,  # Provide speaker name
    language=selected_language,  # Provide language code
    file_path=output_file_path
)
print(f"TTS complete! Check the '{output_folder}' folder for the audio file.")