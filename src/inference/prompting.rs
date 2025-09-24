use serde::Serialize;

#[derive(Serialize)]
pub struct PromptTemplate {
    pub prefix: &'static str,
    pub suffix: &'static str,
}

impl PromptTemplate {
    pub fn apply(&self, text: &str) -> String {
        format!("{}{}{}", self.prefix, text, self.suffix)
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
