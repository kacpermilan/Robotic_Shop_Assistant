# Robotic Shopping Assistant

Robotic Shopping Assistant is a simple project, simulating the hypothetical robotic device, 
that would be capable of helping the customers during the shopping. 
Its primary role is to serve me (repo's author) as a playground to implement various database and
artificial intelligence related functionalities.

Script is currently capable of displaying the camera's video, imitating the robot's vision system,
and with it detect faces (and distinct between known and generic), read barcodes, while displaying
simple overlay interface. Detected barcodes can be added to the cart, both via keyboard input and 
voice interface, allowing the user to phrase the request in any way, and in turn robot would try 
to map it onto one of its commands using LLM. Model will verbally announce its intent to execute 
the command before doing so.

## Technologies used

**Face/barcode recognition**
- opencv
- face_recognition
- pyzbar

**Text-To-Speech**
- Coqui.ai TTS

**Speech-To-Text**
- OpenAI Whisper

**Large Language Model**
- llama_cpp

## Models used

 - **TTS** - tts_models/en/ljspeech/vits
 - **STT** - base.en (whisper)
 - **LLM** - LLaMA 2 7b, downloaded from Meta website and quantized to 4-bit form using llama_cpp