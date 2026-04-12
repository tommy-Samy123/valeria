package com.valeria.app.llm

/**
 * Prepended to the user utterance for Gemma / MediaPipe so replies stay in first-aid scope.
 */
object FirstAidSystemPrompt {

    const val MODEL_ASSET_NAME = "gemma3-270m-it-q8.task"

    fun buildPrompt(userUtterance: String): String {
        // Gemma instruction-tuned model format requires specific control tokens.
        // We wrap the instructions and user query inside the user turn.
        return """<start_of_turn>user
You are Valeria, an AI first-aid responder. Provide actionable, step-by-step first aid instructions immediately based on the user's issue.
Do NOT ask for clarification or more information. Give the best general advice for what was described.
If the situation is severe, tell them to call 911 FIRST, then provide the steps.

${userUtterance.trim()}<end_of_turn>
<start_of_turn>model
""".trimMargin()
    }
}
