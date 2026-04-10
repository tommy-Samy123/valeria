package com.valeria.app.firstaid

/**
 * Simple offline first-aid response engine.
 * Matches user phrases to intent keywords and returns step-by-step instructions.
 * No network, no ML model—lightweight and fast on device.
 */
object FirstAidResponse {

    private val bleedKeywords = listOf(
        "bleed", "bleeding", "blood", "cut", "cut myself", "laceration",
        "wound", "gravel", "fell", "fall", "bike", "accident", "scratch"
    )
    private val burnKeywords = listOf(
        "burn", "burned", "burning", "scald", "hot water", "fire"
    )
    private val chokeKeywords = listOf(
        "choke", "choking", "can't breathe", "something stuck", "throat"
    )
    private val bruiseKeywords = listOf(
        "bruise", "bruised", "bump", "swelling", "hit", "bang"
    )
    private val nosebleedKeywords = listOf(
        "nosebleed", "nose bleed", "nose blood"
    )
    private val sprainKeywords = listOf(
        "sprain", "sprained", "ankle", "wrist", "twist", "twisted", "bone", "break", "broke", "broken", "fracture"
    )
    private val generalHelp = listOf(
        "help", "hurt", "injured", "emergency", "what do i do", "first aid"
    )
    private val greeting = listOf(
        "hi", "hello", "hey", "valeria"
    )

    fun respond(userInput: String): String {
        val normalized = userInput.lowercase().trim()
        if (normalized.isBlank()) return promptToSpeak()

        return when {
            greeting.any { normalized.contains(it) } && normalized.length < 25 ->
                "Hello. I'm Valeria, your first-aid assistant. Say what happened or describe your injury. I'll give you step-by-step instructions. This is not a substitute for emergency care. If it's an emergency, call 911."
            matches(normalized, chokeKeywords) ->
                chokeResponse()
            matches(normalized, bleedKeywords) ->
                bleedingResponse()
            matches(normalized, burnKeywords) ->
                burnResponse()
            matches(normalized, nosebleedKeywords) ->
                nosebleedResponse()
            matches(normalized, sprainKeywords) ->
                sprainResponse()
            matches(normalized, bruiseKeywords) ->
                bruiseResponse()
            matches(normalized, generalHelp) ->
                "I'm here to help. Describe what happened: for example, \"I cut my hand,\" \"I have a nosebleed,\" or \"I burned my arm.\" I'll give you clear steps. In an emergency, call 911."
            else ->
                "I'm not sure I understood. Try saying something like: \"I'm bleeding,\" \"I have a burn,\" or \"I sprained my ankle.\" For life-threatening emergencies, call 911."
        }
    }

    private fun matches(input: String, keywords: List<String>): Boolean =
        keywords.any { input.contains(it) }

    private fun promptToSpeak(): String =
        "I didn't hear that. Tell me what happened or describe your injury in a short sentence."

    private fun bleedingResponse(): String = """
        For bleeding from a cut or scrape: Step 1 — Wash your hands if you can. Step 2 — Rinse the wound with clean, running water. Step 3 — Use a clean cloth or gauze to apply gentle pressure for several minutes until bleeding slows. Step 4 — Apply a thin layer of antibiotic ointment if you have it. Step 5 — Cover with a clean bandage. Change the bandage if it gets wet or dirty. If bleeding doesn't stop after 10 minutes of pressure, or the cut is deep or gaping, get medical help. I'm not a doctor; this is general first-aid guidance.
    """.trimIndent().replace("\n", " ")

    private fun burnResponse(): String = """
        For a minor burn: Step 1 — Cool the burn under cool running water for about 10 minutes. Step 2 — Don't use ice. Step 3 — Gently pat dry. Step 4 — Cover with a sterile non-stick dressing. Don't pop blisters. For serious burns, or burns on the face, hands, or large areas, get medical help right away. If clothing is stuck to the burn, don't remove it; seek emergency care.
    """.trimIndent().replace("\n", " ")

    private fun chokeResponse(): String = """
        If someone is choking and can still cough or breathe, encourage them to keep coughing. If they cannot breathe or make sound: Call 911 immediately. If you're trained, perform abdominal thrusts: stand behind them, make a fist below the ribcage, pull inward and upward. Repeat until the object is out or help arrives. I'm not a substitute for emergency services; call 911 now if someone is choking.
    """.trimIndent().replace("\n", " ")

    private fun nosebleedResponse(): String = """
        For a nosebleed: Step 1 — Sit upright and lean slightly forward. Step 2 — Pinch the soft part of your nose just below the bridge for 10 minutes. Step 3 — Breathe through your mouth. Step 4 — Don't tilt your head back. If bleeding continues after 20 minutes or you feel dizzy, seek medical help.
    """.trimIndent().replace("\n", " ")

    private fun sprainResponse(): String = """
        For a sprain or possible broken bone: remember RICE. R — Rest: avoid using the injured area. I — Ice: apply a cold pack for 15 to 20 minutes every few hours. C — Compression: wrap with an elastic bandage to reduce swelling. E — Elevation: keep the injured area raised above the heart when possible. If the limb looks deformed, you heard a snap, or you cannot move it, seek medical help immediately.
    """.trimIndent().replace("\n", " ")

    private fun bruiseResponse(): String = """
        For a bruise or bump: Step 1 — Rest the area. Step 2 — Apply a cold pack for 15 to 20 minutes several times in the first day. Step 3 — Elevate if possible. Step 4 — After 48 hours, gentle warmth may help. If there's severe pain, swelling, or you hit your head, get medical advice.
    """.trimIndent().replace("\n", " ")
}
