package com.valeria.app.speech

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import java.util.Locale

/**
 * Offline-first speech-to-text using Android's built-in SpeechRecognizer.
 * Prefers on-device recognition when available (API 31+ EXTRA_PREFER_OFFLINE).
 */
class SpeechToTextHelper(
    private val context: Context,
    private val callback: (Result<String>) -> Unit,
    private val onListeningEnded: (() -> Unit)? = null
) {
    private val mainHandler = Handler(Looper.getMainLooper())

    private fun runOnMain(block: () -> Unit) {
        mainHandler.post(block)
    }
    private var speechRecognizer: SpeechRecognizer? = null
    private val recognizerIntent: Intent by lazy {
        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            // Prefer on-device when available (Android 10+ with downloaded language)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            }
        }
    }

    fun startListening() {
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            runOnMain { callback(Result.failure(IllegalStateException("Speech recognition not available on this device"))) }
            runOnMain { onListeningEnded?.invoke() }
            return
        }
        release()
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
            setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(params: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(rmsdB: Float) {}
                override fun onBufferReceived(buffer: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onError(error: Int) {
                    val message = when (error) {
                        SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
                        SpeechRecognizer.ERROR_CLIENT -> "Client error"
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission needed"
                        SpeechRecognizer.ERROR_NETWORK -> "Network error (using offline when possible)"
                        SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
                        SpeechRecognizer.ERROR_NO_MATCH -> "No speech match"
                        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer busy"
                        SpeechRecognizer.ERROR_SERVER -> "Server error"
                        SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech input"
                        else -> "Recognition error: $error"
                    }
                    runOnMain {
                        callback(Result.failure(Exception(message)))
                        onListeningEnded?.invoke()
                    }
                }
                override fun onResults(results: Bundle?) {
                    val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    val text = matches?.firstOrNull()?.trim()
                    runOnMain {
                        if (!text.isNullOrBlank()) {
                            callback(Result.success(text))
                        } else {
                            callback(Result.failure(Exception("No speech recognized")))
                        }
                        onListeningEnded?.invoke()
                    }
                }
                override fun onPartialResults(partialResults: Bundle?) {
                    // Partial results for UI display only; final result comes in onResults
                }
                override fun onEvent(eventType: Int, params: Bundle?) {}
            })
            startListening(recognizerIntent)
        }
    }

    fun stopListening() {
        speechRecognizer?.stopListening()
    }

    fun release() {
        speechRecognizer?.destroy()
        speechRecognizer = null
    }
}
