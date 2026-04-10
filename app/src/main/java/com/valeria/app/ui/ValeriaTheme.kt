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
    primary = Color(0xFFE53935),
    onPrimary = Color.White,
    surface = Color.Black,
    onSurface = Color.White,
    background = Color.Black,
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
