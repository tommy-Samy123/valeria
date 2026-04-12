package com.valeria.app.ui.settings

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RadioButtonDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.valeria.app.llm.FirstAidSystemPrompt
import com.valeria.app.settings.AppSettingsRepository
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    repository: AppSettingsRepository,
    useHardcodedRules: Boolean,
    onBack: () -> Unit
) {
    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings", color = Color.White) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            Icons.Default.ArrowBack,
                            contentDescription = "Back",
                            tint = Color.White
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = Color.Black,
                    navigationIconContentColor = Color.White
                )
            )
        },
        containerColor = Color.Black
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(24.dp)
        ) {
            Text(
                text = "First-aid responses",
                style = MaterialTheme.typography.titleMedium,
                color = Color.White
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Default is Gemma 3 via MediaPipe (on-device). Add ${FirstAidSystemPrompt.MODEL_ASSET_NAME} under app/src/main/assets/ to bundle the model.",
                style = MaterialTheme.typography.bodySmall,
                color = Color.Gray
            )
            Spacer(modifier = Modifier.height(24.dp))

            ResponseModeRow(
                title = "Gemma 3 (MediaPipe)",
                subtitle = "On-device AI — needs a compatible .task model file",
                selected = !useHardcodedRules,
                onClick = { scope.launch { repository.setUseHardcodedRules(false) } }
            )
            Spacer(modifier = Modifier.height(12.dp))
            ResponseModeRow(
                title = "Hardcoded rules",
                subtitle = "Fast keyword matching, no model file",
                selected = useHardcodedRules,
                onClick = { scope.launch { repository.setUseHardcodedRules(true) } }
            )
        }
    }
}

@Composable
private fun ResponseModeRow(
    title: String,
    subtitle: String,
    selected: Boolean,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        RadioButton(
            selected = selected,
            onClick = onClick,
            colors = RadioButtonDefaults.colors(
                selectedColor = MaterialTheme.colorScheme.primary,
                unselectedColor = Color.Gray
            )
        )
        Column(modifier = Modifier.padding(start = 8.dp)) {
            Text(title, color = Color.White, style = MaterialTheme.typography.bodyLarge)
            Text(subtitle, color = Color.Gray, style = MaterialTheme.typography.bodySmall)
        }
    }
}
