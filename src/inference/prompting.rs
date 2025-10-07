use std::path::PathBuf;

use hf_hub::api::sync::Api;
use minijinja::{Environment, context};

use crate::inference::errors::{InferenceError, Result};
use crate::services::llm::ChatRequest;

/*
Some good docs:
- https://huggingface.co/blog/FriendliAI/custom-chat-template
*/

pub trait ChatTemplate: Sync + Send {
    fn format_request(
        &self,
        chat_request: &ChatRequest,
        add_generation_prompt: bool,
        bos_tok: Option<String>,
        eos_tok: Option<String>,
        unk_tok: Option<String>,
    ) -> Result<String>;
}

#[derive(Debug, Clone)]
pub struct SimpleChatTemplate {
    pub prefix: &'static str,
    pub suffix: &'static str,
}

/// Applies the prompt template to the given user input.
///
/// # Arguments
///
/// * `chat_request` - The request from the user to create a chat template
///
/// # Returns
///
/// A `String` containing the formatted prompt, constructed by concatenating
/// the template's prefix, the user input, and the template's suffix.
impl ChatTemplate for SimpleChatTemplate {
    fn format_request(
        &self,
        chat_request: &ChatRequest,
        _add_generation_prompt: bool,
        _bos_tok: Option<String>,
        _eos_tok: Option<String>,
        _unk_tok: Option<String>,
    ) -> Result<String> {
        let mut input = String::new();
        chat_request.messages.iter().for_each(|msg| {
            input.push_str(&format!("{}\n", msg.content));
        });
        Ok(format!("{}{}{}", self.prefix, input, self.suffix))
    }
}

pub const NO_OP: SimpleChatTemplate = SimpleChatTemplate {
    prefix: "",
    suffix: "",
};

pub const INSTRUCT: SimpleChatTemplate = SimpleChatTemplate {
    prefix: "Instruct: ",
    suffix: "\nOutput:",
};

pub const CHAT: SimpleChatTemplate = SimpleChatTemplate {
    prefix: "User: ",
    suffix: "\nAssistant:",
};

pub const QA: SimpleChatTemplate = SimpleChatTemplate {
    prefix: "Question: ",
    suffix: "\nAnswer:",
};

#[derive(Debug, Clone)]
pub struct JinjaChatTemplate<'a> {
    name: String,
    env: Environment<'a>,
}

impl<'a> JinjaChatTemplate<'a> {
    pub fn new(name: impl Into<String>) -> Self {
        let name: String = name.into();
        let mut env = Environment::new();
        // enable python methods such as .strip()
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        Self { name, env }
    }

    /// Creates a JinjaChatTemplate by downloading a template from Hugging Face.
    /// Ensures the downloaded template is properly initialized.
    pub fn from_hf(model_id: String) -> Result<Self> {
        let mut t = Self::new(model_id.clone());
        let path = download_from_hf(model_id.clone())?;
        let source = std::fs::read_to_string(&path).map_err(|e| {
            InferenceError::ChatTemplateError(format!("Failed to read template file: {}", e))
        })?;
        t.env.add_template_owned(model_id, source).unwrap();
        Ok(t)
    }
}

impl<'a> ChatTemplate for JinjaChatTemplate<'a> {
    fn format_request(
        &self,
        chat_request: &ChatRequest,
        add_generation_prompt: bool,
        bos_tok: Option<String>,
        eos_tok: Option<String>,
        unk_tok: Option<String>,
    ) -> Result<String> {
        let tmpl = self.env.get_template(self.name.as_str()).map_err(|e| {
            InferenceError::ChatTemplateError(format!("error fetching template: {}", e))
        })?;
        tmpl.render(context!(
            messages => chat_request.messages,
            add_generation_prompt,
            bos_tok,
            eos_tok,
            unk_tok,
        ))
        .map_err(|e| InferenceError::ChatTemplateError(format!("error rendering template: {}", e)))
    }
}

pub fn download_from_hf(model_id: String) -> Result<PathBuf> {
    let api = Api::new().unwrap();
    let repo = api.model(model_id);
    repo.get("chat_template.jinja").map_err(|e| {
        InferenceError::ChatTemplateError(format!("unable to fetch chat template: {}", e))
    })
}
