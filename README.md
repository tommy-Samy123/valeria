# Valeria

Voice-activated first-aid assistant for Android. Works **offline** using on-device speech and either **Gemma 3 via MediaPipe** (default) or a **hardcoded rule-based** first-aid flow.

## Features

- **Speech-to-text**: Android `SpeechRecognizer` with `EXTRA_PREFER_OFFLINE` when available (no internet required when language pack is installed).
- **Text-to-speech**: Android `TextToSpeech` using the default system engine (offline when language data is installed).
- **Gemma 3 (default)**: On-device LLM using [MediaPipe LLM Inference](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android) (`com.google.mediapipe:tasks-genai`). Requires a compatible **`.task`** model file (see below).
- **Hardcoded rules**: Fast keyword matching for common injuries — no model file; toggle in **Settings**.

## Gemma 3 model file

1. Obtain a **MediaPipe-compatible** Gemma 3 (e.g. 270M IT Q8) model in **`.task`** format, per [Google’s LLM Inference documentation](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android) or community bundles such as [litert-community/gemma-3-270m-it](https://huggingface.co/litert-community/gemma-3-270m-it) on Hugging Face (accept license / `hf auth login` if required).
2. Save the file as **`gemma3-270m-it-q8.task`** (must match this exact name — see `FirstAidSystemPrompt.MODEL_ASSET_NAME`).
3. Place it in **`app/src/main/assets/`** so it is bundled into the APK, **or** copy the same filename into the app’s internal files directory at runtime.

Without this file, **Gemma mode** will show an error when you speak; switch to **Hardcoded rules** in Settings or add the model.

## Requirements

- **minSdk 26** (Android 8.0)
- **targetSdk 34**
- Microphone permission (requested at runtime)
- **Gemma mode**: A recent, capable phone (e.g. Pixel 8 class or Samsung S23+) is recommended; emulators are often unreliable for LLM inference.

### If the app dies while loading Gemma (log shows `lowmemorykiller`, XNNPACK `remap failed`, or `lib/x86_64`)

That usually means **not enough RAM for the model + TensorFlow Lite** (common on **x86_64 AVDs**). Try:

1. **Use a physical Android device** (best).
2. **Increase AVD RAM** (e.g. **8 GB+**), cold boot, and use a **system image with Google Play** if you rely on Play services.
3. **Wipe app data** or uninstall/reinstall to clear a bad `.xnnpack_cache` under app cache.
4. Temporarily switch **Settings → Hardcoded rules** so the rest of the app works.

The app loads the LLM with **GPU first**, then **CPU**, and uses **256 max tokens** to reduce memory use.

For best offline behavior:

- **STT**: On Android 10+, install the desired language for on-device recognition in Settings → Google → Voice.
- **TTS**: Install offline language data for your default language in Settings → Accessibility → Text-to-speech.

## Build and run

The project uses **Android Gradle Plugin 8.5.2+** and **Gradle 8.7** so native libraries (including MediaPipe’s `libllm_inference_engine_jni.so`) are **ZIP-aligned for 16 KB page-size devices**, matching [Google’s requirement](https://developer.android.com/guide/practices/page-sizes) for Play when you target Android 15+.

1. Open the project in **Android Studio** (Hedgehog or newer recommended).
2. Sync Gradle (Android Studio will use or create the Gradle wrapper).
3. Connect a **physical device** (recommended for Gemma) or an emulator (API 26+).
4. Run the **app** configuration.

From the command line (after Gradle wrapper is present):

```bash
# Windows
gradlew.bat assembleDebug

# macOS/Linux
./gradlew assembleDebug
```

Install the APK from `app/build/outputs/apk/debug/`.

## Project structure

```
app/src/main/java/com/valeria/app/
├── MainActivity.kt              # Navigation, chat, mic, Gemma vs hardcoded
├── ValeriaApp.kt
├── firstaid/
│   └── FirstAidResponse.kt      # Rule-based first-aid responses
├── llm/
│   ├── FirstAidSystemPrompt.kt  # Prompt wrapper + MODEL_ASSET_NAME
│   └── GemmaLlmEngine.kt        # MediaPipe LlmInference + session
├── settings/
│   └── AppSettingsRepository.kt # DataStore: hardcoded vs Gemma (default Gemma)
├── speech/
│   ├── SpeechToTextHelper.kt
│   └── TextToSpeechHelper.kt
└── ui/
    ├── ConversationBubble.kt
    ├── ValeriaTheme.kt
    └── settings/
        └── SettingsScreen.kt
```

## Usage

1. Grant microphone permission when prompted.
2. Open **Settings** (gear) to choose **Gemma 3 (MediaPipe)** or **Hardcoded rules**.
3. Tap the **mic** and describe the situation (e.g. *"I cut my hand"*, *"nosebleed"*).
4. Valeria replies in the chat and speaks the answer (Gemma streams text as it generates).

## Disclaimer

Valeria is for **support and quick guidance**, not emergency response. In an emergency, call 911 or local emergency services. Do not substitute Valeria for professional medical care.

## Later (not in this build)

- **Computer vision** for wound/injury analysis when the app is **online** (planned).
