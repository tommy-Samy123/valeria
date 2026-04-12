package com.valeria.app.speech

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import java.util.Locale

/**
 * Speech-to-text using Android's built-in SpeechRecognizer.
 */
class SpeechToTextHelper(
    private val context: Context,
    private val callback: (Result<String>) -> Unit,
    private val getActivelySpeakingText: (() -> String?)? = null,
    private val onSpeechDetected: (() -> Unit)? = null,
    private val onListeningEnded: (() -> Unit)? = null
) {
    private val TAG = "SpeechToTextHelper"
    private val mainHandler = Handler(Looper.getMainLooper())

    private fun runOnMain(block: () -> Unit) {
        if (Looper.myLooper() == Looper.getMainLooper()) {
            block()
        } else {
            mainHandler.post(block)
        }
    }

    private fun isLikelyEcho(recognizedText: String): Boolean {
        val tts = getActivelySpeakingText?.invoke() ?: return false
        if (tts.isBlank() || recognizedText.isBlank()) return false
        
        val ttsWords = tts.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }
        val recWords = recognizedText.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }
        
        if (recWords.isEmpty()) return false
        
        var matchCount = 0
        for (word in recWords) {
            if (ttsWords.contains(word)) matchCount++
        }
        
        val matchRatio = matchCount.toFloat() / recWords.size
        // If 70% or more of the detected words appear in the currently spoken TTS, it's an echo
        return matchRatio >= 0.7f
    }

    private var speechRecognizer: SpeechRecognizer? = null
    
    private val recognizerIntent: Intent by lazy {
        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            // Use Language Tag (e.g. en-US) which is more standard than toString()
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault().toLanguageTag())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            // Tell the recognizer to be a bit more patient
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, 2000L)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, 1500L)
        }
    }

    fun startListening() {
        runOnMain {
            if (!SpeechRecognizer.isRecognitionAvailable(context)) {
                Log.e(TAG, "Speech recognition not available")
                callback(Result.failure(IllegalStateException("Speech recognition not available on this device")))
                onListeningEnded?.invoke()
                return@runOnMain
            }

            if (speechRecognizer == null) {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
                    setRecognitionListener(object : RecognitionListener {
                        override fun onReadyForSpeech(params: Bundle?) {
                            Log.d(TAG, "onReadyForSpeech")
                        }

                        override fun onBeginningOfSpeech() {
                            Log.d(TAG, "onBeginningOfSpeech - Mic started picking up sound")
                        }

                        override fun onRmsChanged(rmsdB: Float) {
                        }

                        override fun onBufferReceived(buffer: ByteArray?) {}

                        override fun onEndOfSpeech() {
                            Log.d(TAG, "onEndOfSpeech")
                        }

                        override fun onError(error: Int) {
                            if (error == SpeechRecognizer.ERROR_NO_MATCH || error == SpeechRecognizer.ERROR_SPEECH_TIMEOUT || error == SpeechRecognizer.ERROR_CLIENT) {
                                // Silent errors for continuous listening loop
                                runOnMain { onListeningEnded?.invoke() }
                                return
                            }
                            val message = when (error) {
                                SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
                                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission needed"
                                SpeechRecognizer.ERROR_NETWORK -> "Network error"
                                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer busy - please wait"
                                SpeechRecognizer.ERROR_SERVER -> "Server error"
                                SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE -> "Language pack missing"
                                else -> "Recognition error: $error"
                            }
                            Log.e(TAG, "onError: $message ($error)")
                            runOnMain {
                                callback(Result.failure(Exception(message)))
                                onListeningEnded?.invoke()
                            }
                        }

                        override fun onResults(results: Bundle?) {
                            val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                            val text = matches?.firstOrNull()?.trim()
                            Log.d(TAG, "onResults: $text")
                            runOnMain {
                                if (!text.isNullOrBlank()) {
                                    if (isLikelyEcho(text)) {
                                        Log.d(TAG, "onResults: Ignored as TTS echo")
                                    } else {
                                        callback(Result.success(text))
                                    }
                                } else {
                                    callback(Result.failure(Exception("No speech recognized")))
                                }
                                onListeningEnded?.invoke()
                            }
                        }

                        override fun onPartialResults(partialResults: Bundle?) {
                            val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                            val partialText = matches?.firstOrNull()?.trim()
                            if (!partialText.isNullOrEmpty() && !isLikelyEcho(partialText)) {
                                runOnMain { onSpeechDetected?.invoke() }
                            }
                            Log.d(TAG, "onPartialResults: $partialText")
                        }

                        override fun onEvent(eventType: Int, params: Bundle?) {}
                    })
                }
            }
            
            Log.d(TAG, "Starting listening with locale: ${Locale.getDefault().toLanguageTag()}")
            speechRecognizer?.startListening(recognizerIntent)
        }
    }

    fun stopListening() {
        runOnMain {
            speechRecognizer?.stopListening()
        }
    }

    fun release() {
        runOnMain {
            speechRecognizer?.destroy()
            speechRecognizer = null
        }
    }
}
