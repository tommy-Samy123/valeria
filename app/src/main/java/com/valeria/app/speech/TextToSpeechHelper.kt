package com.valeria.app.speech

import android.content.Context
import android.os.Build
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale

/**
 * Offline text-to-speech using Android's built-in TTS engine.
 * Uses default system voice; no network required when language data is installed.
 */
class TextToSpeechHelper(
    context: Context,
    private val onReady: () -> Unit = {},
    private val onDone: () -> Unit = {}
) {
    private var tts: TextToSpeech? = null
    private val appContext = context.applicationContext

    fun init() {
        if (tts != null) return
        tts = TextToSpeech(appContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.getDefault()
                tts?.setSpeechRate(0.95f)
                onReady()
            }
        }
    }

    fun speak(text: String, utteranceId: String = "valeria_tts_${System.currentTimeMillis()}") {
        val engine = tts ?: return
        // Remove markdown symbols so TTS doesn't read "asterisk asterisk"
        val cleanText = text.replace(Regex("[*#_`~]"), "")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {}
                override fun onDone(utteranceId: String?) { onDone() }
                override fun onError(utteranceId: String?) { onDone() }
            })
            engine.speak(cleanText, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
        } else {
            @Suppress("DEPRECATION")
            engine.speak(cleanText, TextToSpeech.QUEUE_FLUSH, null)
            onDone()
        }
    }

    fun stop() {
        tts?.stop()
    }

    fun isSpeaking(): Boolean = tts?.isSpeaking == true

    fun release() {
        tts?.stop()
        tts?.shutdown()
        tts = null
    }
}
