use serde::Serialize;

/*
Some good docs:
- https://huggingface.co/blog/FriendliAI/custom-chat-template
*/

#[derive(Serialize, Debug)]
pub struct PromptTemplate {
    pub prefix: &'static str,
    pub suffix: &'static str,
}

/// Applies the prompt template to the given user input.
///
/// # Arguments
///
/// * `user` - The user input string to be embedded within the template.
/// * `_system` - The system prompt string (currently unused).
///
/// # Returns
///
/// A `String` containing the formatted prompt, constructed by concatenating
/// the template's prefix, the user input, and the template's suffix.
impl PromptTemplate {
    pub fn apply(&self, user: &str, _system: &str) -> String {
        format!("{}{}{}", self.prefix, user, self.suffix)
    }
}

pub const NO_OP: PromptTemplate = PromptTemplate {
    prefix: "",
    suffix: "",
};

pub const INSTRUCT: PromptTemplate = PromptTemplate {
    prefix: "Instruct: ",
    suffix: "\nOutput:",
};

pub const CHAT: PromptTemplate = PromptTemplate {
    prefix: "User: ",
    suffix: "\nAssistant:",
};

pub const QA: PromptTemplate = PromptTemplate {
    prefix: "Question: ",
    suffix: "\nAnswer:",
};
