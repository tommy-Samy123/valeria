package com.valeria.app.llm

import android.content.Context
import android.os.Handler
import android.util.Log
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession.LlmInferenceSessionOptions
import com.google.mediapipe.tasks.genai.llminference.ProgressListener
import com.valeria.app.util.isLikelyAndroidEmulator
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors

/**
 * On-device Gemma 3 inference via [MediaPipe LLM Inference](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/android).
 * Expects a converted `.task` model as [FirstAidSystemPrompt.MODEL_ASSET_NAME] in assets (copied to internal storage on first use),
 * or the same filename already under [Context.getFilesDir].
 */
class GemmaLlmEngine(private val context: Context) {

    private val ioExecutor = Executors.newSingleThreadExecutor()
    private var llmInference: LlmInference? = null
    private var session: LlmInferenceSession? = null

    private val sessionOptions: LlmInferenceSessionOptions =
        LlmInferenceSessionOptions.builder()
            .setTopK(40)
            .setTopP(0.95f)
            .setTemperature(0.7f)
            .build()

    fun modelFile(): File = File(context.filesDir, FirstAidSystemPrompt.MODEL_ASSET_NAME)

    fun isModelAvailable(): Boolean = modelFile().exists() && modelFile().length() > 0L

    /**
     * Copies the model from assets if present; otherwise relies on a file the user placed under filesDir.
     */
    fun ensureModelFromAssets(): Result<Unit> {
        val target = modelFile()
        if (target.exists() && target.length() > 0L) return Result.success(Unit)
        return try {
            context.assets.open(FirstAidSystemPrompt.MODEL_ASSET_NAME).use { input ->
                FileOutputStream(target).use { output -> input.copyTo(output) }
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Log.w(TAG, "Model not in assets: ${e.message}")
            if (target.exists() && target.length() == 0L) target.delete()
            Result.failure(e)
        }
    }

    @Synchronized
    fun loadIfNeeded(): Result<Unit> {
        if (llmInference != null) return Result.success(Unit)
        ensureModelFromAssets()
        if (!isModelAvailable()) {
            return Result.failure(
                IllegalStateException(
                    "Gemma model missing. Add ${FirstAidSystemPrompt.MODEL_ASSET_NAME} to app/src/main/assets/ " +
                        "(see README) or copy the same file to app files directory."
                )
            )
        }
        return try {
            val inference = createLlmInferenceWithBackendFallback()
            val newSession = LlmInferenceSession.createFromOptions(inference, sessionOptions)
            llmInference = inference
            session = newSession
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load LlmInference", e)
            close()
            Result.failure(e)
        }
    }

    /**
     * Lower token budget reduces KV / activation memory (helps avoid LMK on constrained devices).
     * Try GPU first: avoids x86_64 emulator XNNPACK mmap issues; falls back to CPU on failure.
     */
    private fun createLlmInferenceWithBackendFallback(): LlmInference {
        val path = modelFile().absolutePath
        val emulator = isLikelyAndroidEmulator()
        if (emulator) {
            Log.w(
                TAG,
                "Running on what looks like an emulator. On-device LLM often needs 8GB+ AVD RAM or a physical device; " +
                    "if the app is killed, use Hardcoded rules or a real phone."
            )
        }
        // CPU first: Avoids FP16 precision issues on Adreno GPUs which cause Detokenizer (-1 vs 0) crashes.
        val order = listOf(LlmInference.Backend.CPU, LlmInference.Backend.GPU)
        var last: Exception? = null
        for (backend in order) {
            try {
                val options = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(path)
                    .setMaxTokens(1024)
                    .setPreferredBackend(backend)
                    .build()
                Log.i(TAG, "Loading LlmInference with backend=$backend maxTokens=1024")
                return LlmInference.createFromOptions(context, options)
            } catch (e: Exception) {
                Log.w(TAG, "LlmInference failed for backend=$backend: ${e.message}")
                last = e
            }
        }
        throw last ?: IllegalStateException("No LlmInference backend succeeded")
    }

    private fun resetSessionLocked() {
        val inference = llmInference ?: return
        session?.close()
        session = LlmInferenceSession.createFromOptions(inference, sessionOptions)
    }

    /**
     * Runs inference on [ioExecutor]. [onPartial] and [onComplete] are posted to [mainHandler].
     */
    fun generateAsync(
        prompt: String,
        mainHandler: Handler,
        onPartial: (String) -> Unit,
        onComplete: (Result<String>) -> Unit
    ) {
        ioExecutor.execute {
            val load = synchronized(this) { loadIfNeeded() }
            load.exceptionOrNull()?.let { e ->
                mainHandler.post { onComplete(Result.failure(e)) }
                return@execute
            }
            val sessionSync = synchronized(this) {
                resetSessionLocked()
                session
            }
            if (sessionSync == null) {
                mainHandler.post {
                    onComplete(Result.failure(IllegalStateException("Session not ready")))
                }
                return@execute
            }
            try {
                sessionSync.addQueryChunk(prompt)
                val accumulated = StringBuilder()
                val listener = object : ProgressListener<String> {
                    override fun run(partialResult: String, done: Boolean) {
                        accumulated.append(partialResult)
                        val text = accumulated.toString()
                        mainHandler.post {
                            onPartial(text)
                            if (done) {
                                onComplete(Result.success(text))
                            }
                        }
                    }
                }
                sessionSync.generateResponseAsync(listener)
            } catch (e: Exception) {
                Log.e(TAG, "generateAsync failed", e)
                mainHandler.post { onComplete(Result.failure(e)) }
            }
        }
    }

    @Synchronized
    fun close() {
        try {
            session?.close()
        } catch (_: Exception) { }
        session = null
        try {
            llmInference?.close()
        } catch (_: Exception) { }
        llmInference = null
    }

    fun shutdown() {
        close()
        ioExecutor.shutdown()
    }

    companion object {
        private const val TAG = "GemmaLlmEngine"
    }
}
