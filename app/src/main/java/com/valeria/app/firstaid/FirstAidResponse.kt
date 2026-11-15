package com.valeria.app.firstaid

/**
 * Simple offline first-aid response engine.
 * Matches user phrases to intent keywords and returns step-by-step instructions.
 * No network, no ML model—lightweight and fast on device.
 *
 * Order of checks matters for safety:
 *   1. Choking (call 911 + abdominal thrusts).
 *   2. Emergency red flags (breathing trouble, unconsciousness, uncontrolled bleeding, ...).
 *   3. Head injury (seek medical care).
 *   4. Routine injuries — every matched injury is answered, most urgent first.
 *   5. Falls with no named injury → ask what hurts.
 *   6. Greeting / general help / fallback.
 *
 * Keywords match whole words only (so "hi" does not match "this" and "hit" does not match "white").
 */
object FirstAidResponse {

    private enum class Injury { BLEEDING, BURN, FRACTURE, SPRAIN, NOSEBLEED, BRUISE }

    private val chokeKeywords = listOf(
        "choke", "choked", "choking", "something stuck", "food stuck", "stuck in my throat",
        "stuck in his throat", "stuck in her throat", "stuck in their throat"
    )

    /** Signs that need emergency services right away instead of routine coaching. */
    private val emergencyKeywords = listOf(
        // Breathing
        "can't breathe", "cant breathe", "cannot breathe", "can not breathe", "not breathing",
        "stopped breathing", "trouble breathing", "difficulty breathing", "hard to breathe",
        "struggling to breathe", "short of breath",
        // Consciousness
        "unconscious", "passed out", "fainted", "not responding", "unresponsive", "won't wake up",
        "wont wake up", "knocked out",
        // Severe bleeding
        "won't stop bleeding", "wont stop bleeding", "can't stop the bleeding", "cant stop the bleeding",
        "bleeding a lot", "bleeding heavily", "heavy bleeding", "spurting", "gushing", "pouring blood",
        "soaked through",
        // Severe trauma
        "bone sticking out", "bone is sticking out", "see the bone", "bone through the skin",
        "stab", "stabbed", "gunshot",
        // Serious burns
        "chemical burn", "electrical burn", "electrocuted", "electric shock", "burned face",
        "burn on my face", "burned my face",
        // Other
        "chest pain", "seizure", "having a seizure"
    )

    private val headInjuryKeywords = listOf(
        "hit my head", "hit his head", "hit her head", "hit their head", "bumped my head",
        "banged my head", "head injury", "hurt my head", "concussion", "fell on my head"
    )

    private val injuryKeywords: Map<Injury, List<String>> = mapOf(
        Injury.BLEEDING to listOf(
            "bleed", "bleeds", "bleeding", "blood", "bloody", "cut", "cuts", "cutting", "cut myself", "laceration",
            "wound", "scrape", "scraped", "scratch", "scratched", "graze", "grazed", "gravel"
        ),
        Injury.BURN to listOf(
            "burn", "burns", "burned", "burnt", "burning", "scald", "scalded", "hot water",
            "hot oil", "fire", "stove"
        ),
        Injury.FRACTURE to listOf(
            "break", "broke", "broken", "fracture", "fractured", "snapped", "heard a snap",
            "heard a crack", "deformed", "bent wrong", "looks bent"
        ),
        Injury.SPRAIN to listOf(
            "sprain", "sprained", "twist", "twisted", "rolled my ankle"
        ),
        Injury.NOSEBLEED to listOf(
            "nosebleed", "nose bleed", "nose bleeding", "nose is bleeding", "bloody nose", "nose blood"
        ),
        Injury.BRUISE to listOf(
            "bruise", "bruised", "bump", "swelling", "swollen", "hit", "bang", "banged"
        )
    )

    /** Most urgent first; also the order responses are read out when several injuries are named. */
    private val injuryPriority = listOf(
        Injury.BLEEDING, Injury.BURN, Injury.FRACTURE, Injury.SPRAIN, Injury.NOSEBLEED, Injury.BRUISE
    )

    /** Joint names alone suggest a sprain only when no more specific injury was named. */
    private val sprainBodyParts = listOf("ankle", "wrist")

    private val fallKeywords = listOf("fell", "fall", "fallen", "tripped", "crashed", "accident", "bike")

    private val generalHelp = listOf(
        "help", "hurt", "injured", "emergency", "what do i do", "first aid"
    )
    private val greeting = listOf(
        "hi", "hello", "hey", "valeria"
    )

    private val regexCache = HashMap<String, Regex>()

    fun respond(userInput: String): String {
        val normalized = normalize(userInput)
        if (normalized.isBlank()) return promptToSpeak()

        if (matches(normalized, chokeKeywords)) return chokeResponse()
        if (matches(normalized, emergencyKeywords)) return emergencyResponse()
        if (matches(normalized, headInjuryKeywords)) return headInjuryResponse()

        val injuries = detectInjuries(normalized)
        if (injuries.isNotEmpty()) {
            if (injuries.size == 1) return responseFor(injuries.first())
            return "You mentioned more than one injury. I'll go through them one at a time. " +
                injuries.joinToString(" Next: ") { responseFor(it) }
        }

        return when {
            matches(normalized, fallKeywords) ->
                "It sounds like you had a fall. Where does it hurt, and is anything bleeding, burned, swollen, or bent the wrong way? If you hit your head, passed out, or can't move a limb, call 911."
            matches(normalized, greeting) && normalized.length < 25 ->
                "Hello. I'm Valeria, your first-aid assistant. Say what happened or describe your injury. I'll give you step-by-step instructions. This is not a substitute for emergency care. If it's an emergency, call 911."
            matches(normalized, generalHelp) ->
                "I'm here to help. Describe what happened: for example, \"I cut my hand,\" \"I have a nosebleed,\" or \"I burned my arm.\" I'll give you clear steps. In an emergency, call 911."
            else ->
                "I'm not sure I understood. Try saying something like: \"I'm bleeding,\" \"I have a burn,\" or \"I sprained my ankle.\" For life-threatening emergencies, call 911."
        }
    }

    private fun detectInjuries(input: String): List<Injury> {
        val found = injuryPriority.filter { matches(input, injuryKeywords.getValue(it)) }.toMutableList()
        if (found.isEmpty() && matches(input, sprainBodyParts)) found.add(Injury.SPRAIN)
        // A suspected break must not also get sprain (RICE-only) advice.
        if (Injury.FRACTURE in found) found.remove(Injury.SPRAIN)
        // "Bloody nose" should get nosebleed steps, not wound care — unless a separate wound is named.
        if (Injury.NOSEBLEED in found && Injury.BLEEDING in found) {
            val withoutNose = injuryKeywords.getValue(Injury.NOSEBLEED)
                .fold(input) { acc, kw -> acc.replace(wordRegex(kw), " ") }
            if (!matches(withoutNose, injuryKeywords.getValue(Injury.BLEEDING))) found.remove(Injury.BLEEDING)
        }
        // "Hit" / "banged" alone counts as a bruise only when nothing more specific was named.
        if (found.size > 1 && Injury.BRUISE in found &&
            !matches(input, listOf("bruise", "bruised", "bump", "swelling", "swollen"))
        ) found.remove(Injury.BRUISE)
        return found
    }

    private fun responseFor(injury: Injury): String = when (injury) {
        Injury.BLEEDING -> bleedingResponse()
        Injury.BURN -> burnResponse()
        Injury.FRACTURE -> fractureResponse()
        Injury.SPRAIN -> sprainResponse()
        Injury.NOSEBLEED -> nosebleedResponse()
        Injury.BRUISE -> bruiseResponse()
    }

    /** Lowercases, unifies curly apostrophes, and strips punctuation other than apostrophes. */
    private fun normalize(input: String): String =
        input.lowercase()
            .replace('’', '\'')
            .replace(Regex("[^a-z0-9' ]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()

    private fun wordRegex(keyword: String): Regex =
        regexCache.getOrPut(keyword) { Regex("(?<![a-z0-9'])" + Regex.escape(keyword) + "(?![a-z0-9'])") }

    private fun matches(input: String, keywords: List<String>): Boolean =
        keywords.any { wordRegex(it).containsMatchIn(input) }

    private fun promptToSpeak(): String =
        "I didn't hear that. Tell me what happened or describe your injury in a short sentence."

    private fun emergencyResponse(): String = """
        This may be a medical emergency. Call 911 now, or ask someone nearby to call. If there is heavy bleeding, press firmly on the wound with a clean cloth and don't let go. If the person is not breathing and you are trained, start CPR. Stay with them until help arrives. I can't handle emergencies; emergency services can.
    """.trimIndent().replace("\n", " ")

    private fun headInjuryResponse(): String = """
        For a head injury: Call 911 if the person passed out, is confused, is vomiting, has a seizure, has unequal pupils, or has clear fluid from the nose or ears. Otherwise: Step 1 — Sit or lie down and stay still. Step 2 — Hold a cold pack wrapped in cloth on the bump for 15 to 20 minutes. Step 3 — Don't be alone for the next 24 hours, and get checked by a doctor. Watch for a worsening headache, drowsiness, or confusion.
    """.trimIndent().replace("\n", " ")

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

    private fun fractureResponse(): String = """
        For a possible broken bone: this needs to be checked by a doctor. Step 1 — Don't try to straighten or move the injured part. Step 2 — Keep it still and supported in the position you found it, using a pillow, rolled towel, or sling. Step 3 — Put a cold pack wrapped in cloth on it for up to 20 minutes. Step 4 — Get medical care today. Call 911 if the bone is poking through the skin, the limb is cold, blue, or numb, or the injury is to the head, neck, back, hip, or thigh.
    """.trimIndent().replace("\n", " ")

    private fun sprainResponse(): String = """
        For a sprain: remember RICE. R — Rest: avoid using the injured area. I — Ice: apply a cold pack for 15 to 20 minutes every few hours. C — Compression: wrap with an elastic bandage to reduce swelling. E — Elevation: keep the injured area raised above the heart when possible. If the limb looks deformed, you heard a snap, you can't put weight on it, or you cannot move it, seek medical help.
    """.trimIndent().replace("\n", " ")

    private fun bruiseResponse(): String = """
        For a bruise or bump: Step 1 — Rest the area. Step 2 — Apply a cold pack for 15 to 20 minutes several times in the first day. Step 3 — Elevate if possible. Step 4 — After 48 hours, gentle warmth may help. If there's severe pain, swelling, or you hit your head, get medical advice.
    """.trimIndent().replace("\n", " ")
}
