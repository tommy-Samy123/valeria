plugins {
    // 8.5.1+ ZIP-aligns uncompressed JNI libs for 16 KB page-size devices (Play requirement Nov 2025)
    id("com.android.application") version "8.5.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.24" apply false
}
