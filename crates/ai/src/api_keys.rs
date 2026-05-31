pub use crate::aws_credentials::{AwsCredentials, AwsCredentialsState};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use warp_multi_agent_api as api;
use warpui::{Entity, ModelContext, SingletonEntity};
use warpui_extras::secure_storage::{self, AppContextExt};

const SECURE_STORAGE_KEY: &str = "AiApiKeys";

/// Emitted when user-provided API keys are updated in-memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApiKeyManagerEvent {
    KeysUpdated,
}

/// User-provided API keys for AI providers.
///
/// These are used for "Bring Your Own API Key" functionality, allowing
/// users to use their own API keys instead of Warp's.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiKeys {
    pub google: Option<String>,
    pub anthropic: Option<String>,
    pub openai: Option<String>,
    pub open_router: Option<String>,
    pub groq: Option<String>,
    pub nvidia_nim: Option<String>,
    pub ollama_enabled: Option<bool>,
    pub custom_endpoints: Vec<CustomEndpoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CustomEndpoint {
    pub name: String,
    pub url: String,
    pub api_key: String,
    pub models: Vec<CustomEndpointModel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CustomEndpointModel {
    pub name: String,
    pub alias: Option<String>,
    /// Stable identifier used as `ModelConfig.{base,coding,cli_agent,computer_use_agent}` and
    /// as the `CustomModelProviders.providers[*].models[*].config_key` on the request wire.
    /// Generated as a UUIDv4 at model creation.
    pub config_key: String,
}

impl CustomEndpointModel {
    /// Picker label: prefer the user-provided alias; fall back to the raw model name
    /// so a row is never blank.
    pub fn display_label(&self) -> &str {
        match self.alias.as_deref() {
            Some(alias) if !alias.trim().is_empty() => alias,
            _ => &self.name,
        }
    }
}

impl ApiKeys {
    pub fn has_any_key(&self) -> bool {
        self.openai.is_some()
            || self.anthropic.is_some()
            || self.google.is_some()
            || self.open_router.is_some()
            || self.groq.is_some()
            || self.nvidia_nim.is_some()
            || self
                .custom_endpoints
                .iter()
                .any(|endpoint| !endpoint.api_key.trim().is_empty())
    }

    pub fn has_nvidia(&self) -> bool {
        self.nvidia_nim.as_ref().is_some_and(|k| !k.trim().is_empty())
    }

    pub fn has_groq(&self) -> bool {
        self.groq.as_ref().is_some_and(|k| !k.trim().is_empty())
    }

    pub fn nvidia_nim_key(&self) -> Option<&str> {
        self.nvidia_nim.as_deref().filter(|k| !k.trim().is_empty())
    }

    pub fn groq_key(&self) -> Option<&str> {
        self.groq.as_deref().filter(|k| !k.trim().is_empty())
    }

    /// Returns `true` when the user has at least one custom endpoint configured.
    pub fn has_custom_endpoints(&self) -> bool {
        !self.custom_endpoints.is_empty()
    }
}

/// Controls how AWS credentials are refreshed by [`ApiKeyManager`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AwsCredentialsRefreshStrategy {
    /// Load credentials from the local AWS credential chain (~/.aws). This is the default.
    #[default]
    LocalChain,
    /// Credentials are managed externally via OIDC/STS.
    /// The task ID is used to scope the STS AssumeRoleWithWebIdentity session.
    /// The role ARN + region are the info used to assume the IAM role via STS.
    OidcManaged {
        task_id: Option<String>,
        role_arn: String,
        region: String,
    },
}

/// A structure that manages API keys for AI providers.
pub struct ApiKeyManager {
    keys: ApiKeys,
    pub(crate) aws_credentials_state: AwsCredentialsState,
    aws_credentials_refresh_strategy: AwsCredentialsRefreshStrategy,
    secure_storage_write_version: u64,
}

impl ApiKeyManager {
    pub fn new(ctx: &mut ModelContext<Self>) -> Self {
        let keys = Self::load_keys_from_secure_storage(ctx);
        Self {
            keys,
            aws_credentials_state: AwsCredentialsState::Missing,
            aws_credentials_refresh_strategy: AwsCredentialsRefreshStrategy::default(),
            secure_storage_write_version: 0,
        }
    }

    pub fn keys(&self) -> &ApiKeys {
        &self.keys
    }

    pub fn set_google_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.google = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_anthropic_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.anthropic = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_openai_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.openai = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_open_router_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.open_router = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_groq_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.groq = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_nvidia_nim_key(&mut self, key: Option<String>, ctx: &mut ModelContext<Self>) {
        self.keys.nvidia_nim = key;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_ollama_enabled(&mut self, enabled: Option<bool>, ctx: &mut ModelContext<Self>) {
        self.keys.ollama_enabled = enabled;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn add_custom_endpoint(
        &mut self,
        name: String,
        url: String,
        api_key: String,
        models: Vec<(String, Option<String>, Option<String>)>,
        ctx: &mut ModelContext<Self>,
    ) {
        self.keys.custom_endpoints.push(CustomEndpoint {
            name,
            url,
            api_key,
            models: models
                .into_iter()
                .map(|(name, alias, config_key)| CustomEndpointModel {
                    name,
                    alias,
                    config_key: config_key
                        .filter(|k| !k.is_empty())
                        .unwrap_or_else(|| Uuid::new_v4().to_string()),
                })
                .collect(),
        });
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn save_custom_endpoint(
        &mut self,
        index: usize,
        name: String,
        url: String,
        api_key: String,
        models: Vec<(String, Option<String>, Option<String>)>,
        ctx: &mut ModelContext<Self>,
    ) {
        if index >= self.keys.custom_endpoints.len() {
            return;
        }
        self.keys.custom_endpoints[index] = CustomEndpoint {
            name,
            url,
            api_key,
            models: models
                .into_iter()
                .map(|(name, alias, config_key)| CustomEndpointModel {
                    name,
                    alias,
                    config_key: config_key
                        .filter(|k| !k.is_empty())
                        .unwrap_or_else(|| Uuid::new_v4().to_string()),
                })
                .collect(),
        };
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn remove_custom_endpoint(&mut self, index: usize, ctx: &mut ModelContext<Self>) {
        if index >= self.keys.custom_endpoints.len() {
            return;
        }
        self.keys.custom_endpoints.remove(index);
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn clear_custom_endpoints(&mut self, ctx: &mut ModelContext<Self>) {
        if self.keys.custom_endpoints.is_empty() {
            return;
        }
        self.keys.custom_endpoints.clear();
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
        self.write_keys_to_secure_storage(ctx);
    }

    pub fn set_aws_credentials_state(
        &mut self,
        state: AwsCredentialsState,
        ctx: &mut ModelContext<Self>,
    ) {
        self.aws_credentials_state = state;
        ctx.emit(ApiKeyManagerEvent::KeysUpdated);
    }

    pub fn aws_credentials_state(&self) -> &AwsCredentialsState {
        &self.aws_credentials_state
    }

    pub fn aws_credentials_refresh_strategy(&self) -> AwsCredentialsRefreshStrategy {
        self.aws_credentials_refresh_strategy.clone()
    }

    pub fn set_aws_credentials_refresh_strategy(
        &mut self,
        strategy: AwsCredentialsRefreshStrategy,
    ) {
        self.aws_credentials_refresh_strategy = strategy;
    }

    /// Builds the `CustomModelProviders` registry that ships with every agent request.
    ///
    /// Emits one [`CustomModelProvider`] per configured [`CustomEndpoint`], each populated with
    /// all of its [`CustomEndpointModel`]s. The per-model `config_key` is what the server uses
    /// to map a `ModelConfig.{base,coding,cli_agent,computer_use_agent}` selection back to a
    /// user-provided endpoint, so it MUST be the same UUID we store locally.
    ///
    /// Returns `None` when custom models should not be included or no endpoint has both a
    /// non-empty URL and API key.
    pub fn custom_model_providers_for_request(
        &self,
        include_custom_models: bool,
    ) -> Option<api::request::settings::CustomModelProviders> {
        let mut providers: Vec<_> = Vec::new();

        // User-defined custom endpoints — gated by include_custom_models flag
        if include_custom_models {
            providers.extend(
                self.keys
                    .custom_endpoints
                    .iter()
                    .filter(|endpoint| !endpoint.url.trim().is_empty() && !endpoint.api_key.is_empty())
                    .map(
                        |endpoint| api::request::settings::custom_model_providers::CustomModelProvider {
                            base_url: endpoint.url.clone(),
                            api_key: endpoint.api_key.clone(),
                            models: endpoint
                                .models
                                .iter()
                                .filter(|m| !m.name.trim().is_empty() && !m.config_key.is_empty())
                                .map(
                                    |m| api::request::settings::custom_model_providers::CustomModel {
                                        slug: m.name.clone(),
                                        config_key: m.config_key.clone(),
                                    },
                                )
                                .collect(),
                        },
                    )
                    .filter(|provider| !provider.models.is_empty()),
            );
        }

        // Add Groq as custom model provider if user has Groq API key
        if let Some(key) = self.keys.groq.as_ref() {
            if !key.trim().is_empty() {
                let models: Vec<_> = Self::groq_llm_models()
                    .iter()
                    .map(|(n, k)| api::request::settings::custom_model_providers::CustomModel {
                        slug: (*n).to_string(),
                        config_key: (*k).to_string(),
                    })
                    .collect();
                if !models.is_empty() {
                    providers.push(api::request::settings::custom_model_providers::CustomModelProvider {
                        base_url: "https://api.groq.com/v1".to_string(),
                        api_key: key.clone(),
                        models,
                    });
                }
            }
        }

        // Add NVIDIA NIM as custom model provider if user has NVIDIA API key
        if let Some(key) = self.keys.nvidia_nim.as_ref() {
            if !key.trim().is_empty() {
                let models: Vec<_> = Self::nvidia_llm_models()
                    .iter()
                    .map(|(n, k)| api::request::settings::custom_model_providers::CustomModel {
                        slug: (*n).to_string(),
                        config_key: (*k).to_string(),
                    })
                    .collect();
                if !models.is_empty() {
                    providers.push(api::request::settings::custom_model_providers::CustomModelProvider {
                        base_url: "https://integrate.api.nvidia.com/v1".to_string(),
                        api_key: key.clone(),
                        models,
                    });
                }
            }
        }

        // Add Ollama as custom model provider if enabled
        if self.keys.ollama_enabled.unwrap_or(false) {
            let models: Vec<_> = Self::ollama_llm_models()
                .iter()
                .map(|(n, k)| api::request::settings::custom_model_providers::CustomModel {
                    slug: (*n).to_string(),
                    config_key: (*k).to_string(),
                })
                .collect();
            if !models.is_empty() {
                providers.push(api::request::settings::custom_model_providers::CustomModelProvider {
                    base_url: "http://localhost:11434/v1".to_string(),
                    api_key: "".to_string(),
                    models,
                });
            }
        }

        if providers.is_empty() {
            None
        } else {
            Some(api::request::settings::CustomModelProviders { providers })
        }
    }

    fn groq_llm_models() -> Vec<(&'static str, &'static str)> {
        vec![
            ("groq/llama-3.3-70b-versatile", "groq-llama-3-3-70b-versatile"),
            ("groq/llama-3.1-70b-versatile", "groq-llama-3-1-70b-versatile"),
            ("groq/llama-3.1-8b-instant", "groq-llama-3-1-8b-instant"),
            ("groq/llama3-70b", "groq-llama3-70b"),
            ("groq/llama3-8b", "groq-llama3-8b"),
            ("groq/mixtral-8x7b", "groq-mixtral-8x7b"),
            ("groq/gemma2-9b", "groq-gemma2-9b"),
            ("groq/gemma-7b", "groq-gemma-7b"),
        ]
    }

    fn nvidia_llm_models() -> Vec<(&'static str, &'static str)> {
        vec![
            ("nvidia/llama-3.3-70b-versatile", "nvidia-llama-3-3-70b-versatile"),
            ("nvidia/llama-3.1-70b-instruct", "nvidia-llama-3-1-70b-instruct"),
            ("nvidia/llama-3.1-8b-instruct", "nvidia-llama-3-1-8b-instruct"),
            ("nvidia/llama-3-70b-instruct", "nvidia-llama-3-70b-instruct"),
            ("nvidia/llama-3-8b-instruct", "nvidia-llama-3-8b-instruct"),
            ("nvidia/mixtral-8x7b-v3", "nvidia-mixtral-8x7b-v3"),
            ("nvidia/mixtral-8x7b-instruct", "nvidia-mixtral-8x7b-instruct"),
            ("nvidia/mistral-large", "nvidia-mistral-large"),
            ("nvidia/mistral-7b-instruct-v3", "nvidia-mistral-7b-instruct-v3"),
            ("nvidia/nemotron-4-340b-instruct", "nvidia-nemotron-4-340b-instruct"),
            ("nvidia/gemma2-27b-it", "nvidia-gemma2-27b-it"),
            ("nvidia/gemma2-9b-it", "nvidia-gemma2-9b-it"),
            ("nvidia/gemma-7b-it", "nvidia-gemma-7b"),
            ("nvidia/deepseek-r1-distill-llama-70b", "nvidia-deepseek-r1-distill-llama-70b"),
            ("nvidia/deepseek-r1-distill-llama-8b", "nvidia-deepseek-r1-distill-llama-8b"),
            ("nvidia/qwen2.5-72b-instruct", "nvidia-qwen2-5-72b-instruct"),
            ("nvidia/qwen2.5-32b-instruct", "nvidia-qwen2-5-32b-instruct"),
            ("nvidia/qwen2.5-14b-instruct", "nvidia-qwen2-5-14b-instruct"),
            ("nvidia/qwen2.5-7b-instruct", "nvidia-qwen2-5-7b-instruct"),
            ("nvidia/phi-4-mini-instruct", "nvidia-phi-4-mini-instruct"),
            ("nvidia/phi-3.5-mini-instruct", "nvidia-phi-3-5-mini-instruct"),
        ]
    }

    fn ollama_llm_models() -> Vec<(&'static str, &'static str)> {
        vec![
            ("llama3.3:70b", "ollama-llama3-3-70b"),
            ("llama3.1:70b", "ollama-llama3-1-70b"),
            ("llama3.1:8b", "ollama-llama3-1-8b"),
            ("llama3.1:latest", "ollama-llama3-1-latest"),
            ("llama3:70b", "ollama-llama3-70b"),
            ("llama3:8b", "ollama-llama3-8b"),
            ("llama2:70b", "ollama-llama2-70b"),
            ("llama2:13b", "ollama-llama2-13b"),
            ("mixtral:8x22b", "ollama-mixtral-8x22b"),
            ("mixtral:8x7b", "ollama-mixtral-8x7b"),
            ("mistral:latest", "ollama-mistral-latest"),
            ("mistral:7b", "ollama-mistral-7b"),
            ("gemma3:27b", "ollama-gemma3-27b"),
            ("gemma3:12b", "ollama-gemma3-12b"),
            ("gemma2:27b", "ollama-gemma2-27b"),
            ("gemma2:12b", "ollama-gemma2-12b"),
            ("gemma2:9b", "ollama-gemma2-9b"),
            ("gemma:7b", "ollama-gemma-7b"),
            ("qwen2.5:72b", "ollama-qwen2-5-72b"),
            ("qwen2.5:32b", "ollama-qwen2-5-32b"),
            ("qwen2.5:14b", "ollama-qwen2-5-14b"),
            ("qwen2.5:7b", "ollama-qwen2-5-7b"),
            ("qwen2.5:3b", "ollama-qwen2-5-3b"),
            ("qwen2.5:1.5b", "ollama-qwen2-5-1-5b"),
            ("deepseek-r1:70b", "ollama-deepseek-r1-70b"),
            ("deepseek-r1:32b", "ollama-deepseek-r1-32b"),
            ("deepseek-r1:14b", "ollama-deepseek-r1-14b"),
            ("deepseek-r1:8b", "ollama-deepseek-r1-8b"),
            ("deepseek-r1:1.5b", "ollama-deepseek-r1-1-5b"),
            ("phi4:14b", "ollama-phi4-14b"),
            ("phi4-mini:3.8b", "ollama-phi4-mini-3-8b"),
            ("phi3.5:latest", "ollama-phi3-5-latest"),
            ("nemotron:70b", "ollama-nemotron-70b"),
            ("wizardlm2:70b", "ollama-wizardlm2-70b"),
            ("wizardlm2:8x22b", "ollama-wizardlm2-8x22b"),
            ("codellama:70b", "ollama-codellama-70b"),
            ("codellama:13b", "ollama-codellama-13b"),
            ("codellama:7b", "ollama-codellama-7b"),
            ("codegemma:22b", "ollama-codegemma-22b"),
            ("codegemma:7b", "ollama-codegemma-7b"),
            ("llava:13b", "ollama-llava-13b"),
            ("llava:7b", "ollama-llava-7b"),
            ("llava-llama3:8b", "ollama-llava-llama3-8b"),
            ("granite:8b", "ollama-granite-8b"),
            ("granite:20b", "ollama-granite-20b"),
            ("hermes3:70b", "ollama-hermes3-70b"),
            ("command-r7b", "ollama-command-r7b"),
            ("command-r35b", "ollama-command-r35b"),
        ]
    }

    pub fn api_keys_for_request(
        &self,
        include_byo_keys: bool,
        include_aws_bedrock_credentials: bool,
    ) -> Option<api::request::settings::ApiKeys> {
        let anthropic = include_byo_keys
            .then(|| self.keys.anthropic.clone())
            .flatten()
            .unwrap_or_default();
        let openai = include_byo_keys
            .then(|| self.keys.openai.clone())
            .flatten()
            .unwrap_or_default();
        let google = include_byo_keys
            .then(|| self.keys.google.clone())
            .flatten()
            .unwrap_or_default();
        let open_router = include_byo_keys
            .then(|| self.keys.open_router.clone())
            .flatten()
            .unwrap_or_default();

        // Also include credentials when running with OIDC-managed Bedrock inference, regardless
        // of the per-user setting flag (which only applies to the local credential chain path).
        let include_aws = include_aws_bedrock_credentials
            || matches!(
                self.aws_credentials_refresh_strategy,
                AwsCredentialsRefreshStrategy::OidcManaged { .. }
            );
        let aws_credentials = include_aws
            .then(|| match self.aws_credentials_state {
                AwsCredentialsState::Loaded {
                    ref credentials, ..
                } => Some(credentials.clone().into()),
                _ => None,
            })
            .flatten();

        if anthropic.is_empty()
            && openai.is_empty()
            && google.is_empty()
            && open_router.is_empty()
            && aws_credentials.is_none()
        {
            None
        } else {
            Some(api::request::settings::ApiKeys {
                anthropic,
                openai,
                google,
                open_router,
                allow_use_of_warp_credits: false,
                aws_credentials,
            })
        }
    }

    fn load_keys_from_secure_storage(ctx: &mut ModelContext<Self>) -> ApiKeys {
        let key_json = match ctx.secure_storage().read_value(SECURE_STORAGE_KEY) {
            Ok(json) => json,
            Err(e) => {
                if !matches!(e, secure_storage::Error::NotFound) {
                    log::error!("Failed to read API keys from secure storage: {e:#}");
                }
                return ApiKeys::default();
            }
        };

        match serde_json::from_str(&key_json) {
            Ok(keys) => keys,
            Err(e) => {
                log::error!("Failed to deserialize API keys: {e:#}");
                ApiKeys::default()
            }
        }
    }

    fn write_keys_to_secure_storage(&mut self, ctx: &mut ModelContext<Self>) {
        let json = match serde_json::to_string(&self.keys) {
            Ok(json) => json,
            Err(e) => {
                log::error!("Failed to serialize API keys: {e:#}");
                return;
            }
        };
        self.secure_storage_write_version += 1;
        let write_version = self.secure_storage_write_version;

        // Defer the keychain write so it doesn't block the current event
        // processing. The in-memory state is already updated and events
        // already emitted, so the UI updates immediately while the
        // potentially slow platform secure-storage call runs in a
        // subsequent main-thread callback. Skip stale callbacks so older
        // writes cannot complete after and overwrite a newer payload.
        ctx.spawn(async move { json }, move |me, json, ctx| {
            if write_version != me.secure_storage_write_version {
                return;
            }
            if let Err(e) = ctx.secure_storage().write_value(SECURE_STORAGE_KEY, &json) {
                log::error!("Failed to write API keys to secure storage: {e:#}");
            }
        });
    }
}

impl Entity for ApiKeyManager {
    type Event = ApiKeyManagerEvent;
}

impl SingletonEntity for ApiKeyManager {}

#[cfg(test)]
#[path = "api_keys_tests.rs"]
mod tests;
