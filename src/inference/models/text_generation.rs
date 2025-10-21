use crate::inference::{
    errors::{InferenceError, Result},
    inputs::{InputBuilder, ModelOutput, OutputBuilder},
    models::config::{GenerationConfig, ModelConfig},
};
use log::{debug, trace};
use ort::session::Session;
use rand::Rng;
use serde_json::Value;

pub struct TextGenerationModel {
    pub name: String,
    pub generate_cfg: GenerationConfig,
    session: Session,
    input_builder: InputBuilder,
    output_builder: OutputBuilder,
}

/// Implementation of the `TextGenerationModel` struct, providing methods for model initialization and inference.
///
/// # Methods
///
/// - `new(cfg: ModelConfig) -> Result<Self>`
///   Constructs a new `TextGenerationModel` instance using the provided configuration.
///   Initializes the model session, retrieves metadata (name, description, version), and sets up input/output builders.
///   Returns an error if session creation or metadata retrieval fails.
///
/// - `run<T: PrimitiveTensorElementType + Copy>(&mut self, tokens: &[i64]) -> Result<ModelOutput<T>>`
///   Runs inference on the model using the provided token sequence.
///   Builds input values, executes the session, and extracts the first output.
///   Returns the model output or an error if inference fails.
impl TextGenerationModel {
    pub fn new(cfg: ModelConfig) -> Result<Self> {
        let name = cfg.model_name.clone();
        let session = cfg.clone().build_session()?;

        // Access metadata and signatures BEFORE creating the model to avoid borrowing issues
        let metadata = session.metadata().map_err(|e| {
            InferenceError::ModelMetadataError(format!("Failed to get metadata: {}", e))
        })?;
        let metadata_name = metadata.name().map_err(|e| {
            InferenceError::ModelMetadataError(format!("Failed to get model name: {}", e))
        })?;
        let description = metadata.description().map_err(|e| {
            InferenceError::ModelMetadataError(format!("Failed to get model description: {}", e))
        })?;
        let version = metadata.version().map_err(|e| {
            InferenceError::ModelMetadataError(format!("Failed to get model version: {}", e))
        })?;

        // Debug output
        let banner = "*".repeat(41);
        debug!("{}{}", banner, banner);
        debug!("Name: {}", metadata_name);
        debug!("Description: {}", description);
        debug!("Version: {}", version);
        let input_builder = InputBuilder::from_session_inputs(&session.inputs);
        let output_builder = OutputBuilder::from_session_outputs(&session.outputs);
        debug!("{}{}", banner, banner);

        let model_cfg = cfg.model_cfg.clone();
        let session = cfg.build_session()?;
        let generate_cfg = model_cfg.get_generation_config()?;

        Ok(Self {
            session,
            generate_cfg,
            input_builder,
            output_builder,
            name,
        })
    }

    pub fn generate(
        &mut self,
        mut input_tokens: Vec<i64>,
        max_tokens: i32,
        min_p: f32,
        temperature: f32,
        top_k: i64,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        let mut generated_tokens = Vec::new();
        let mut rng = rand::rng();
        for _ in 0..max_tokens {
            let output = self.run(&input_tokens)?;
            let logits = output.logits()?;
            let tokens = process_logits(logits, temperature, top_p)?;
            if tokens.is_empty() {
                break;
            }
            let top_k_size = top_k.min(tokens.len() as i64);
            let selected = tokens[rng.random_range(0..top_k_size as usize)];
            let token = selected.0 as i64;
            let probability = selected.1;
            if probability < min_p {
                debug!(
                    "Improbable token generated (token={}, prob={}, count={})",
                    token,
                    probability,
                    generated_tokens.len()
                );
                // continue;
            }

            if end_of_sequence(token, &self.generate_cfg.eos_token_id) {
                debug!("EOS found on token {}", generated_tokens.len());
                break;
            }
            input_tokens.push(token);
            generated_tokens.push(token as u32);
        }
        Ok(generated_tokens)
    }

    fn run(&mut self, tokens: &[i64]) -> Result<ModelOutput<f32>> {
        let input_values = self.input_builder.build(tokens)?;
        match self.session.run(input_values) {
            Ok(r) => self.output_builder.get_first(r),
            Err(e) => Err(InferenceError::OnnxRuntime(e)),
        }
    }
}

fn end_of_sequence(token: i64, eos_token: &Option<serde_json::Value>) -> bool {
    if let Some(eos_token) = &eos_token {
        match eos_token {
            Value::Array(data) => data.iter().any(|v| v.as_i64() == Some(token)),
            Value::Number(val) => val.as_i64() == Some(token),
            _ => false,
        }
    } else {
        false
    }
}

fn process_logits(
    logits: Vec<(usize, f32)>,
    temperature: f32,
    top_p: f32,
) -> Result<Vec<(usize, f32)>> {
    // Temperature scaling:
    // Divide logits by temperature before softmax.
    // T < 1.0 → sharper distribution (more deterministic).
    // T > 1.0 → flatter distribution (more random).
    // Ref: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig.temperature
    let inv_temp = 1.0 / temperature;

    // Numerical stability trick:
    // Subtract the maximum logit to avoid overflow in exp().
    // Ref: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    let max_logit = logits
        .iter()
        .map(|(_, logit)| *logit * inv_temp)
        .fold(f32::NEG_INFINITY, f32::max);

    // Apply exp((logit / T) - max_logit) for each token.
    // This transforms logits into positive unnormalized values.
    // Ref: https://en.wikipedia.org/wiki/Softmax_function
    let exp_vals: Vec<(usize, f32)> = logits
        .iter()
        .map(|(id, logit)| {
            let scaled = (*logit * inv_temp) - max_logit;
            (*id, scaled.exp())
        })
        .collect();

    // Normalize to probabilities: p_i = exp_i / Σ exp_j
    // Ensures probabilities sum to 1.
    let sum: f32 = exp_vals.iter().map(|(_, val)| *val).sum();
    let mut probs: Vec<(usize, f32)> = exp_vals
        .into_iter()
        .map(|(id, val)| (id, val / sum))
        .collect();

    // Sort tokens by probability (descending) so top-k or top-p
    // filtering can be applied easily downstream.
    // Ref: https://huggingface.co/blog/how-to-generate
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

    // Apply top-p (nucleus) filtering if requested
    // Value of 1.0 means we are not filtering.
    if top_p == 1.0 {
        return Ok(probs);
    }

    if 0.0 < top_p && top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut cutoff = 0;
        for &(_, p) in &probs {
            // Always return >=1 elements
            if cumulative + p > top_p && cutoff > 0 {
                break;
            }
            cumulative += p;
            cutoff += 1;
        }
        probs.truncate(cutoff);

        // Optional: renormalize after filtering so probs sum to 1.0
        let sum_p: f32 = probs.iter().map(|&(_, p)| p).sum();
        if sum_p > 0.0 {
            for (_, p) in probs.iter_mut() {
                *p /= sum_p;
            }
        }
        trace!("{:?}", probs);
    } else {
        return Err(InferenceError::TextGenerationError(format!(
            "Invalid top-p parameter '{}'",
            top_p
        )));
    }

    Ok(probs)
}
