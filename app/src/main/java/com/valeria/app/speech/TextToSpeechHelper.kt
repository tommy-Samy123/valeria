package com.valeria.app.speech

import android.content.Context
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.io.File
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
    private var mediaPlayer: MediaPlayer? = null
    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    private var isPlaybackActive = false
    private var isPaused = false

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
        
        stop() // Stop previous playback if any
        
        val cleanText = text.replace(Regex("[*#_`~]"), "")
        val tempFile = File(appContext.cacheDir, "tts_temp.wav")
        if (tempFile.exists()) tempFile.delete()
        
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) {
                mainHandler.post { playFile(tempFile) }
            }
            override fun onError(utteranceId: String?) {
                mainHandler.post { onDone() }
            }
        })
        
        val params = Bundle().apply {
            putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, utteranceId)
        }
        engine.synthesizeToFile(cleanText, params, tempFile, utteranceId)
    }

    private fun playFile(file: File) {
        try {
            mediaPlayer = MediaPlayer().apply {
                setDataSource(file.absolutePath)
                setOnCompletionListener {
                    isPlaybackActive = false
                    isPaused = false
                    onDone()
                }
                setOnErrorListener { _, _, _ ->
                    isPlaybackActive = false
                    isPaused = false
                    onDone()
                    true
                }
                prepare()
                start()
            }
            isPlaybackActive = true
            isPaused = false
        } catch (e: Exception) {
            e.printStackTrace()
            onDone()
        }
    }

    fun pause() {
        mediaPlayer?.let {
            if (it.isPlaying) {
                it.pause()
                isPaused = true
                isPlaybackActive = false
            }
        }
    }

    fun resume() {
        mediaPlayer?.let {
            if (!it.isPlaying && isPaused) {
                it.start()
                isPaused = false
                isPlaybackActive = true
            }
        }
    }

    fun stop() {
        tts?.stop()
        mediaPlayer?.let {
            if (it.isPlaying) it.stop()
            it.release()
        }
        mediaPlayer = null
        isPlaybackActive = false
        isPaused = false
    }

    fun isSpeaking(): Boolean = isPlaybackActive

    fun isPausedState(): Boolean = isPaused

    fun release() {
        stop()
        tts?.shutdown()
        tts = null
    }
}
