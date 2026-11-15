package com.valeria.app.firstaid

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FirstAidResponseTest {

    private fun reply(input: String) = FirstAidResponse.respond(input)

    private fun isEmergency(r: String) = r.startsWith("This may be a medical emergency")
    private fun isFracture(r: String) = r.contains("possible broken bone")
    private fun isSprain(r: String) = r.contains("remember RICE")
    private fun isBleeding(r: String) = r.contains("bleeding from a cut or scrape")
    private fun isBurn(r: String) = r.contains("For a minor burn")
    private fun isNosebleed(r: String) = r.contains("For a nosebleed")
    private fun isHeadInjury(r: String) = r.contains("For a head injury")
    private fun isGreeting(r: String) = r.startsWith("Hello. I'm Valeria")
    private fun isChoking(r: String) = r.contains("abdominal thrusts")

    // Bug 1: a suspected fracture used to get sprain/RICE advice.
    @Test fun brokenBoneGetsFractureAdviceNotRice() {
        for (input in listOf("I think I broke my wrist", "my arm is broken", "possible fracture in my ankle")) {
            val r = reply(input)
            assertTrue(input, isFracture(r))
            assertFalse(input, isSprain(r))
        }
    }

    @Test fun sprainStillGetsRice() {
        assertTrue(isSprain(reply("I sprained my ankle")))
        assertTrue(isSprain(reply("I twisted my wrist")))
        assertTrue(isSprain(reply("my ankle hurts")))
    }

    // Bug 2: "can't breathe" used to be treated as choking instead of an emergency.
    @Test fun breathingTroubleIsEmergency() {
        for (input in listOf("I can't breathe", "he's not breathing", "she is having trouble breathing", "I can’t breathe")) {
            assertTrue(input, isEmergency(reply(input)))
        }
    }

    @Test fun otherRedFlagsAreEmergency() {
        for (input in listOf("he passed out", "it won't stop bleeding", "there's bone sticking out", "chemical burn on my arm")) {
            assertTrue(input, isEmergency(reply(input)))
        }
    }

    @Test fun chokingStillGetsChokingSteps() {
        assertTrue(isChoking(reply("my son is choking")))
    }

    // Bug 3: "fell", "bike", "accident" used to get bleeding advice.
    @Test fun fallWithoutInjuryAsksWhatHurts() {
        for (input in listOf("I fell off my bike", "there was an accident", "I fell down")) {
            val r = reply(input)
            assertFalse(input, isBleeding(r))
            assertTrue(input, r.contains("Where does it hurt"))
        }
    }

    @Test fun headInjuryFromFallGetsHeadInjuryAdvice() {
        val r = reply("I fell off my bike and hit my head")
        assertTrue(isHeadInjury(r))
        assertFalse(isBleeding(r))
    }

    @Test fun fallWithScrapeGetsWoundCare() {
        assertTrue(isBleeding(reply("I fell and scraped my knee")))
    }

    // Bug 4: substring matching ("hi" in "this", "hit" in "white", "cut" in "execute").
    @Test fun keywordsMatchWholeWordsOnly() {
        assertFalse(isGreeting(reply("this is weird")))
        assertFalse(isBleeding(reply("execute the plan")))
        assertFalse(reply("white shirt").contains("bruise or bump"))
        assertFalse(isBurn(reply("I have heartburn")))
    }

    @Test fun greetingStillWorks() {
        assertTrue(isGreeting(reply("hi")))
        assertTrue(isGreeting(reply("Hey Valeria")))
    }

    @Test fun greetingDoesNotHideInjury() {
        assertTrue(isBleeding(reply("hey I cut my hand")))
    }

    // Bug 5: only the first matched injury was answered.
    @Test fun multipleInjuriesAreAllAnswered() {
        val r = reply("I burned my hand and now it's bleeding")
        assertTrue(r.startsWith("You mentioned more than one injury"))
        assertTrue(isBleeding(r))
        assertTrue(isBurn(r))
        assertTrue(r.indexOf("bleeding from a cut") < r.indexOf("For a minor burn"))
    }

    @Test fun bloodyNoseIsNosebleedOnly() {
        val r = reply("I have a bloody nose")
        assertTrue(isNosebleed(r))
        assertFalse(isBleeding(r))
    }

    @Test fun blankInputPromptsToSpeak() {
        assertTrue(reply("   ").startsWith("I didn't hear that"))
    }
}
