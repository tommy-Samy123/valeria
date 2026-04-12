package com.valeria.app.util

import android.os.Build

/**
 * Heuristic for Android Studio / Genymotion / QEMU AVDs.
 * On-device LLM + XNNPACK on x86_64 emulators often hits mmap issues and LMK; prefer a physical device.
 */
fun isLikelyAndroidEmulator(): Boolean {
    val fp = Build.FINGERPRINT
    val model = Build.MODEL
    val man = Build.MANUFACTURER
    val prod = Build.PRODUCT
    val hw = Build.HARDWARE
    val brand = Build.BRAND
    return fp.startsWith("generic")
        || fp.startsWith("unknown")
        || fp.contains("emulator")
        || fp.contains("test-keys")
        || model.contains("google_sdk", ignoreCase = true)
        || model.contains("Emulator", ignoreCase = true)
        || model.contains("Android SDK built for x86", ignoreCase = true)
        || man.contains("Genymotion", ignoreCase = true)
        || hw.contains("goldfish", ignoreCase = true)
        || hw.contains("ranchu", ignoreCase = true)
        || prod.contains("sdk", ignoreCase = true)
        || prod.contains("emulator", ignoreCase = true)
        || prod.contains("simulator", ignoreCase = true)
        || (brand.startsWith("generic") && deviceIsGeneric())
}

private fun deviceIsGeneric(): Boolean {
    val dev = Build.DEVICE
    return dev.startsWith("generic") || dev.contains("generic_x86", ignoreCase = true)
}
