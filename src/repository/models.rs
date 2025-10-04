use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Transaction {
    pub id: Option<i64>,
    pub prompt: Option<String>,
    pub response: Option<String>,
    pub model_name: Option<String>,
}

impl Transaction {
    // Need a better name for this - want to start from prompt and set the response later
    // This is the default constructor for a request
    pub fn new(prompt: String) -> Self {
        Self {
            id: None,
            prompt: Some(prompt),
            response: None,
            model_name: None,
        }
    }

    pub fn set_response(&mut self, resp: String) {
        self.response = Some(resp)
    }

    pub fn set_model(&mut self, model: String) {
        self.model_name = Some(model)
    }
}
