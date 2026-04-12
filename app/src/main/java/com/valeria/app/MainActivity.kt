package com.valeria.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
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
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.valeria.app.firstaid.FirstAidResponse
import com.valeria.app.llm.FirstAidSystemPrompt
import com.valeria.app.llm.GemmaLlmEngine
import com.valeria.app.settings.AppSettingsRepository
import com.valeria.app.speech.SpeechToTextHelper
import com.valeria.app.speech.TextToSpeechHelper
import com.valeria.app.ui.ConversationBubble
import com.valeria.app.ui.ValeriaTheme
import com.valeria.app.ui.settings.SettingsScreen

private const val ROUTE_HOME = "home"
private const val ROUTE_SETTINGS = "settings"

class MainActivity : ComponentActivity() {

    private var micPermissionGranted by mutableStateOf(false)

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        micPermissionGranted = granted
        if (!granted) {
            Toast.makeText(this, "Microphone permission is required for voice commands", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        micPermissionGranted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

        setContent {
            ValeriaTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    ValeriaRoot(
                        hasMicPermission = micPermissionGranted,
                        onRequestPermission = {
                            permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                        }
                    )
                }
            }
        }
    }
}

@Composable
fun ValeriaRoot(
    hasMicPermission: Boolean,
    onRequestPermission: () -> Unit
) {
    val context = LocalContext.current
    val appContext = context.applicationContext
    val navController = rememberNavController()
    val settingsRepo = remember { AppSettingsRepository(appContext) }
    val gemmaEngine = remember { GemmaLlmEngine(appContext) }
    val mainHandler = remember { Handler(Looper.getMainLooper()) }
    val useHardcodedRules by settingsRepo.useHardcodedRules.collectAsStateWithLifecycle(initialValue = false)

    DisposableEffect(Unit) {
        onDispose { gemmaEngine.shutdown() }
    }

    LaunchedEffect(useHardcodedRules) {
        if (useHardcodedRules) {
            gemmaEngine.close()
        }
    }

    NavHost(
        navController = navController,
        startDestination = ROUTE_HOME
    ) {
        composable(ROUTE_HOME) {
            ValeriaScreen(
                hasMicPermission = hasMicPermission,
                onRequestPermission = onRequestPermission,
                useHardcodedRules = useHardcodedRules,
                gemmaEngine = gemmaEngine,
                mainHandler = mainHandler,
                onOpenSettings = { navController.navigate(ROUTE_SETTINGS) }
            )
        }
        composable(ROUTE_SETTINGS) {
            SettingsScreen(
                repository = settingsRepo,
                useHardcodedRules = useHardcodedRules,
                onBack = { navController.popBackStack() }
            )
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ValeriaScreen(
    hasMicPermission: Boolean,
    onRequestPermission: () -> Unit,
    useHardcodedRules: Boolean,
    gemmaEngine: GemmaLlmEngine,
    mainHandler: Handler,
    onOpenSettings: () -> Unit,
    viewModel: ValeriaViewModel = androidx.lifecycle.viewmodel.compose.viewModel()
) {
    val context = LocalContext.current
    val messages = viewModel.messages
    var isListening by remember { mutableStateOf(false) }
    var isSpeaking by remember { mutableStateOf(false) }
    var isGenerating by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()

    val ttsHelperState = remember { mutableStateOf<TextToSpeechHelper?>(null) }
    val useHardcodedNow by rememberUpdatedState(useHardcodedRules)

    val stt = remember(context) {
        SpeechToTextHelper(
            context,
            callback = { result ->
                result.onSuccess { text ->
                    if (text.isBlank()) return@onSuccess
                    if (isGenerating) return@onSuccess

                    messages.add(ConversationMessage(text, fromUser = true))

                    if (useHardcodedNow) {
                        val response = FirstAidResponse.respond(text)
                        messages.add(ConversationMessage(response, fromUser = false))
                    } else {
                        messages.add(ConversationMessage("Thinking…", fromUser = false))
                        val assistantIndex = messages.lastIndex
                        isGenerating = true
                        val prompt = FirstAidSystemPrompt.buildPrompt(text)
                        gemmaEngine.generateAsync(
                            prompt = prompt,
                            mainHandler = mainHandler,
                            onPartial = { partial ->
                                if (assistantIndex in messages.indices) {
                                    messages[assistantIndex] = ConversationMessage(partial, fromUser = false)
                                }
                            },
                            onComplete = { res ->
                                isGenerating = false
                                res.onSuccess { full ->
                                    if (assistantIndex in messages.indices) {
                                        messages[assistantIndex] = ConversationMessage(full, fromUser = false)
                                    }
                                    if (full.isNotBlank()) {
                                        ttsHelperState.value?.speak(full)
                                        isSpeaking = true
                                    }
                                }
                                res.onFailure { e ->
                                    val err = e.message ?: "Gemma inference failed"
                                    if (assistantIndex in messages.indices) {
                                        messages[assistantIndex] = ConversationMessage(err, fromUser = false)
                                    }
                                    Toast.makeText(context, err, Toast.LENGTH_LONG).show()
                                }
                            }
                        )
                    }
                }.onFailure { error ->
                    Toast.makeText(context, error.message ?: "Speech error", Toast.LENGTH_SHORT).show()
                }
            },
            onListeningEnded = { isListening = false }
        )
    }

    DisposableEffect(Unit) {
        val tts = TextToSpeechHelper(
            context,
            onDone = { isSpeaking = false }
        )
        tts.init()
        ttsHelperState.value = tts
        onDispose {
            stt.release()
            tts.release()
            ttsHelperState.value = null
        }
    }

    LaunchedEffect(messages.size, useHardcodedRules) {
        if (messages.isEmpty()) return@LaunchedEffect
        scrollState.animateScrollTo(scrollState.maxValue)
        if (!useHardcodedRules) return@LaunchedEffect
        val last = messages.last()
        if (!last.fromUser && last.text.isNotBlank()) {
            ttsHelperState.value?.speak(last.text)
            isSpeaking = true
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("Valeria", color = Color.White)
                        Text(
                            text = if (useHardcodedRules) "Hardcoded rules" else "Gemma 3 · MediaPipe",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.Gray
                        )
                    }
                },
                actions = {
                    IconButton(onClick = onOpenSettings) {
                        Icon(
                            Icons.Default.Settings,
                            contentDescription = "Settings",
                            tint = Color.White
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface,
                    titleContentColor = Color.White
                )
            )
        },
        containerColor = MaterialTheme.colorScheme.background
    ) { padding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.background)
                .padding(padding)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp)
            ) {
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
                        modifier = Modifier.padding(vertical = 16.dp),
                        textAlign = TextAlign.Center
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
                            modifier = Modifier
                                .padding(vertical = 24.dp)
                                .fillMaxWidth(),
                            textAlign = TextAlign.Center
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
                            if (isGenerating) {
                                Toast.makeText(
                                    context,
                                    "Please wait for the current reply to finish.",
                                    Toast.LENGTH_SHORT
                                ).show()
                                return@FloatingActionButton
                            }
                            if (isSpeaking) {
                                ttsHelperState.value?.stop()
                                isSpeaking = false
                            }
                            if (isListening) {
                                stt.stopListening()
                                isListening = false
                            } else {
                                isListening = true
                                stt.startListening()
                            }
                        },
                        containerColor = MaterialTheme.colorScheme.primary,
                        contentColor = Color.White,
                        modifier = Modifier.size(72.dp)
                    ) {
                        Icon(
                            imageVector = if (isListening) Icons.Default.MicNone else Icons.Default.Mic,
                            contentDescription = if (isListening) "Listening…" else "Tap to speak",
                            modifier = Modifier.size(32.dp)
                        )
                    }
                }
            }
        }
    }
}

class ValeriaViewModel : androidx.lifecycle.ViewModel() {
    val messages = androidx.compose.runtime.mutableStateListOf<ConversationMessage>()
}

data class ConversationMessage(
    val text: String,
    val fromUser: Boolean
)
