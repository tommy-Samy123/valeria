package com.valeria.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.font.FontWeight

@Composable
fun ConversationBubble(
    text: String,
    fromUser: Boolean,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .widthIn(max = 320.dp)
            .background(
                brush = if (fromUser) Brush.horizontalGradient(
                    colors = listOf(MaterialTheme.colorScheme.primary, MaterialTheme.colorScheme.secondary)
                ) else Brush.horizontalGradient(
                    colors = listOf(MaterialTheme.colorScheme.surface, MaterialTheme.colorScheme.surface.copy(alpha = 0.8f))
                ),
                shape = if (fromUser) RoundedCornerShape(20.dp, 20.dp, 4.dp, 20.dp) 
                        else RoundedCornerShape(20.dp, 20.dp, 20.dp, 4.dp)
            )
            .padding(16.dp)
    ) {
        Text(
            text = parseMarkdown(text),
            style = MaterialTheme.typography.bodyMedium,
            color = Color.White
        )
    }
}

fun parseMarkdown(text: String): AnnotatedString {
    val normalizedText = text.replace("\\n", "\n")
    return buildAnnotatedString {
        val parts = normalizedText.split("**")
        for (i in parts.indices) {
            if (i % 2 == 1) {
                withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(parts[i]) }
            } else {
                append(parts[i])
            }
        }
    }
}
