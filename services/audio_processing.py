import threading
import io
import wave
import numpy as np
import av
from openai import AzureOpenAI

class AudioFrameBuffer:
    """Buffers audio frames from streamlit-webrtc for processing."""
    def __init__(self):
        self.frames_lock = threading.Lock()
        self.frames = []
        self.sample_rate = 48000  # WebRTC default
        self.channels = 1
        self.sample_width = 2  # 16-bit PCM

    def add_frame(self, frame: av.AudioFrame):
        with self.frames_lock:
            self.frames.append(frame.to_ndarray())

    def get_wav_bytes(self) -> bytes:
        with self.frames_lock:
            if not self.frames:
                return b""
            
            sound_chunk = np.concatenate(self.frames, axis=1)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.sample_rate)
                wf.writeframes(sound_chunk.tobytes())
            
            wav_bytes = buffer.getvalue()
            self.frames = []  # Clear buffer after processing
            return wav_bytes

def transcribe_audio_with_openai(audio_bytes: bytes, api_key: str, endpoint: str, api_version: str, whisper_deployment: str) -> str:
    """Transcribes audio using the Azure OpenAI Whisper model."""
    if not all([api_key, endpoint, api_version, whisper_deployment]):
        return "Error: Azure OpenAI credentials for Whisper are not configured."
    
    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav" # Required by the API

        transcript = client.audio.transcriptions.create(
            model=whisper_deployment,
            file=audio_file,
            response_format="text"
        )
        return transcript
    except Exception as e:
        return f"Transcription exception: {e}"
