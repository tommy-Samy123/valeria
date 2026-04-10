package com.valeria.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicNone
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.valeria.app.firstaid.FirstAidResponse
import com.valeria.app.speech.SpeechToTextHelper
import com.valeria.app.speech.TextToSpeechHelper
import com.valeria.app.ui.ConversationBubble
import com.valeria.app.ui.ValeriaTheme

class MainActivity : ComponentActivity() {

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        micPermissionGranted = granted
    }

    private var micPermissionGranted = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            micPermissionGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
            if (!micPermissionGranted) {
                permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        } else {
            micPermissionGranted = true
        }

        setContent {
            ValeriaTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color.Black
                ) {
                    ValeriaScreen(
                        hasMicPermission = micPermissionGranted,
                        onRequestPermission = {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                                permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                            }
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun ValeriaScreen(
    hasMicPermission: Boolean,
    onRequestPermission: () -> Unit
) {
    val context = LocalContext.current
    val messages = remember { mutableStateListOf<ConversationMessage>() }
    val isListening = remember { mutableStateOf(false) }
    val isSpeaking = remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()

    val stt = remember {
        SpeechToTextHelper(
            context,
            callback = { result ->
                result.onSuccess { text ->
                    if (text.isNotBlank()) {
                        messages.add(ConversationMessage(text, fromUser = true))
                        val response = FirstAidResponse.respond(text)
                        messages.add(ConversationMessage(response, fromUser = false))
                    }
                }
            },
            onListeningEnded = { isListening.value = false }
        )
    }

    val tts = remember {
        TextToSpeechHelper(
            context,
            onDone = { isSpeaking.value = false }
        )
    }

    DisposableEffect(Unit) {
        tts.init()
        onDispose {
            stt.release()
            tts.release()
        }
    }

    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty() && messages.last().fromUser == false) {
            tts.speak(messages.last().text)
            isSpeaking.value = true
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
        ) {
            Text(
                text = "Valeria",
                style = MaterialTheme.typography.headlineMedium,
                color = Color.White,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center
            )
            Text(
                text = "First-aid assistant · Works offline",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(16.dp))

            if (!hasMicPermission) {
                Text(
                    text = "Microphone access is needed so Valeria can hear you. Tap the mic to allow.",
                    color = Color.White.copy(alpha = 0.8f),
                    modifier = Modifier.padding(vertical = 16.dp)
                )
            }

            Column(
                modifier = Modifier
                    .weight(1f)
                    .verticalScroll(scrollState),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                if (messages.isEmpty()) {
                    Text(
                        text = "Say \"Valeria, I cut my hand\" or describe what happened. Tap the mic and speak.",
                        color = Color.White.copy(alpha = 0.6f),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(vertical = 24.dp)
                    )
                } else {
                    messages.forEach { msg ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = if (msg.fromUser) Arrangement.End else Arrangement.Start
                        ) {
                            ConversationBubble(
                                text = msg.text,
                                fromUser = msg.fromUser
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            Box(
                modifier = Modifier.fillMaxWidth(),
                contentAlignment = Alignment.Center
            ) {
                FloatingActionButton(
                    onClick = {
                        if (!hasMicPermission) {
                            onRequestPermission()
                            return@FloatingActionButton
                        }
                        if (isSpeaking.value) {
                            tts.stop()
                            isSpeaking.value = false
                        }
                        if (isListening.value) {
                            stt.stopListening()
                            isListening.value = false
                        } else {
                            isListening.value = true
                            stt.startListening()
                        }
                    },
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = Color.White,
                    modifier = Modifier.size(72.dp)
                ) {
                    Icon(
                        imageVector = if (isListening.value) Icons.Default.MicNone else Icons.Default.Mic,
                        contentDescription = if (isListening.value) "Listening…" else "Tap to speak",
                        modifier = Modifier.size(32.dp)
                    )
                }
            }
        }
    }
}

private data class ConversationMessage(
    val text: String,
    val fromUser: Boolean
)
