- 👋 Hi, I’m @amanjnc
- 👀 I’m interested in graphics design and coding..
- 🌱 I’m currently learning AAiT ..
- 💞️ I’m looking to collaborate on coding...
- 📫 How to reach me amanuelbeyene662@gmail.com...

<!---
amanjnc/amanjnc is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
import sounddevice as sd
from scipy.io.wavfile import write
from speechbrain.inference import EncoderASR

# Set the audio parameters
sample_rate = 16000  # Sample rate of the audio
duration = 10       # Duration of the audio recording in seconds

# Record audio from the microphone
print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()  # Wait until recording is complete

# Save the recorded audio to a file
write("live_audio.wav", sample_rate, audio)

# Load the pre-trained ASR model
asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-dvoice-amharic", savedir="pretrained_models/asr-wav2vec2-dvoice-amharic")

# Transcribe the recorded audio
transcription = asr_model.transcribe_file("live_audio.wav")
print("Transcription:", transcription)
