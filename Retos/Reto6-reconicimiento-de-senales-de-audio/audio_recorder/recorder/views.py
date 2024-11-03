# recorder/views.py
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import pyaudio
import wave
import threading
import os
import numpy as np
import librosa
from scipy.stats import skew
import joblib

# Variables globales para la configuración de audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio_data = b''
audio = None
recording = False
output_path = "media/recording.wav"

def index(request):
    return render(request, 'recorder/index.html')

def start_recording(request):
    global audio_data, audio, recording
    audio_data = b''
    recording = True
    threading.Thread(target=record_audio).start()
    return JsonResponse({'status': 'Recording started'})

def record_audio():
    global audio_data, audio, recording
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while recording:
            data = stream.read(CHUNK)
            audio_data += data
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def stop_recording(request):
    global recording
    recording = False
    save_audio()
    result = process_audio(output_path)
    return JsonResponse({'status': 'Recording stopped', 'result': result})

# def save_audio():
#     global audio_data
#     if not os.path.exists(output_path):
#         os.makedirs(os.path.dirname(output_path))
#     with wave.open(output_path, 'wb') as wav_file:
#         wav_file.setnchannels(CHANNELS)
#         wav_file.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
#         wav_file.setframerate(RATE)
#         wav_file.writeframes(audio_data)

def save_audio():
    global audio_data
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path))
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wav_file.setframerate(RATE)
        wav_file.writeframes(audio_data)

def process_audio(file_path):
    x, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
    X_New = np.resize(x, 1000000)
    audio = np.array(X_New)
    Total_caracteristicas = 7
    Matriz_caracteristica = np.zeros((1, Total_caracteristicas))
    Modelo_entrenado = joblib.load('D:/ADMIN/Music/Audios/Modelo_Audio.pkl')
    Varianza = np.var(audio)
    Desviacion = np.std(audio)
    rms_amplitude = np.sqrt(np.mean(np.square(audio)))
    zero_crossings = np.where(np.diff(np.sign(audio)))[0]
    zcr = len(zero_crossings)
    skewness = skew(audio)
    Magnitud, phase = librosa.magphase(librosa.stft(audio))
    RMS_vector = librosa.feature.rms(S=Magnitud)
    RMS = RMS_vector.mean()
    Times_vector = librosa.times_like(RMS_vector)
    Times = Times_vector.mean()
    Matriz_caracteristica[0, 0] = Varianza
    Matriz_caracteristica[0, 1] = Desviacion
    Matriz_caracteristica[0, 2] = rms_amplitude
    Matriz_caracteristica[0, 3] = zcr
    Matriz_caracteristica[0, 4] = skewness
    Matriz_caracteristica[0, 5] = RMS
    Matriz_caracteristica[0, 6] = Times
    prediccion = Modelo_entrenado.predict(Matriz_caracteristica)
    Etapa = ['Adultos', 'Ancianos', 'Infantes', 'Jóvenes']
    resultado =f'Perteneces al grupo de los {Etapa[int(prediccion[0])-1]}'
    return resultado
