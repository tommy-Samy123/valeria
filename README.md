# Valeria

Voice-activated first-aid assistant for Android. Works **offline** using on-device speech and a simple rule-based first-aid flow.

## Features

- **Speech-to-text**: Android `SpeechRecognizer` with `EXTRA_PREFER_OFFLINE` when available (no internet required when language pack is installed).
- **Text-to-speech**: Android `TextToSpeech` using the default system engine (offline when language data is installed).
- **First-aid flow**: Rule-based responses for bleeding, burns, choking, nosebleeds, sprains, bruises, and general help. No cloud or ML model required.

## Requirements

- **minSdk 26** (Android 8.0)
- **targetSdk 34**
- Microphone permission (requested at runtime)

For best offline behavior:

- **STT**: On Android 10+, install the desired language for on-device recognition in Settings → Google → Voice.
- **TTS**: Install offline language data for your default language in Settings → Accessibility → Text-to-speech.

## Build and run

1. Open the project in **Android Studio** (Hedgehog or newer recommended).
2. Sync Gradle (Android Studio will use or create the Gradle wrapper).
3. Connect a device or start an emulator (API 26+).
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
├── MainActivity.kt          # Compose UI, permission, mic button
├── ValeriaApp.kt            # Application class
├── firstaid/
│   └── FirstAidResponse.kt   # Offline rule-based first-aid responses
├── speech/
│   ├── SpeechToTextHelper.kt # Offline-first STT (SpeechRecognizer)
│   └── TextToSpeechHelper.kt# Offline TTS
└── ui/
    ├── ConversationBubble.kt
    └── ValeriaTheme.kt
```

## Usage

1. Grant microphone permission when prompted.
2. Tap the **mic** button and say something like:
   - *"Valeria, I cut my hand"*
   - *"I have a nosebleed"*
   - *"I burned my arm"*
   - *"I sprained my ankle"*
3. Valeria replies with step-by-step first-aid instructions and speaks them aloud.

## Disclaimer

Valeria is for **support and quick guidance**, not emergency response. In an emergency, call 911 or local emergency services. Do not substitute Valeria for professional medical care.

## Later (not in this build)

- **Computer vision** for wound/injury analysis when the app is **online** (planned).
- Optional on-device small model for richer conversation (e.g. TensorFlow Lite) on capable devices.
