use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Transaction {
    pub id: i64,
    pub prompt: String,
    pub response: String,
}
