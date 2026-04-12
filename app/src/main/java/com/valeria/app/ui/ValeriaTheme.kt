package com.valeria.app.ui

import android.app.Activity
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.SideEffect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalView
import androidx.core.view.WindowCompat

private val ValeriaDarkScheme = darkColorScheme(
    primary = Color(0xFF6C63FF), // Vibrant indigo/purple
    secondary = Color(0xFF00C9FF), // Cyan/Blue
    tertiary = Color(0xFFB388FF),
    onPrimary = Color.White,
    surface = Color(0xFF1A1A2E), // Deep midnight blue for cards
    onSurface = Color.White,
    background = Color(0xFF10101A), // Darker midnight for background
    onBackground = Color.White
)

@Composable
fun ValeriaTheme(
    content: @Composable () -> Unit
) {
    val colorScheme = ValeriaDarkScheme
    val view = LocalView.current
    if (!view.isInEditMode) {
        SideEffect {
            val window = (view.context as Activity).window
            window.statusBarColor = Color.Black.toArgb()
            window.navigationBarColor = Color.Black.toArgb()
            WindowCompat.getInsetsController(window, view).isAppearanceLightStatusBars = false
        }
    }
    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
