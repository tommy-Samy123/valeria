package com.valeria.app.settings

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "valeria_settings")

class AppSettingsRepository(private val context: Context) {

    private val useHardcodedRulesKey = booleanPreferencesKey("use_hardcoded_rules")

    /**
     * When false (default), responses use Gemma 3 via MediaPipe on-device LLM.
     * When true, responses use the built-in rule-based first-aid matcher.
     */
    val useHardcodedRules: Flow<Boolean> = context.dataStore.data.map { prefs ->
        prefs[useHardcodedRulesKey] ?: false
    }

    suspend fun setUseHardcodedRules(enabled: Boolean) {
        context.dataStore.edit { prefs ->
            prefs[useHardcodedRulesKey] = enabled
        }
    }
}
