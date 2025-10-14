use core::num;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Error, Result as GenericResult};
use log::{debug, trace};
use ndarray::Array;
use ort::tensor::PrimitiveTensorElementType;
use ort::{
    session::{Session, SessionOutputs},
    value::Value,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::Value as JSONValue;

use crate::inference::{
    cache::KVCache,
    errors::{InferenceError, Result},
    model::{GenerationConfig, ModelConfig},
};


pub struct AutoRegressiveModel {
    pub name: String,
    pub generate_cfg: GenerationConfig,
    session: Session,
    rng: Arc<Mutex<StdRng>>,
    cache: KVCache,
}

impl AutoRegressiveModel {
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
        debug!("{}{}", banner, banner);
        debug!("Model inputs:");
        for input in &session.inputs {
            debug!(
                "  {} [{:?}] shape={:?}",
                input.name,
                input.input_type.tensor_type().unwrap(),
                input.input_type.tensor_shape().unwrap()
            )
        }

        debug!("Model outputs:");
        for output in &session.outputs {
            debug!(
                "  {} [{:?}] shape={:?}",
                output.name,
                output.output_type.tensor_type().unwrap(),
                output.output_type.tensor_shape().unwrap()
            )
        }

        let model_cfg = cfg.model_cfg.clone();
        let session = cfg.build_session()?;
        let generate_cfg = model_cfg.get_generation_config()?;

        Ok(Self {
            session,
            generate_cfg,
            name,
            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(42))),
            cache: KVCache::default(),
        })
    }

    fn create_input_tensors(
        &mut self,
        input_ids: &[i64],
        use_cache: bool,
    ) -> GenericResult<Vec<(String, Value)>, Error> {
        // TODO: Not a huge fan tbh
        let batch_size = 1i64;
        let seq_len = input_ids.len() as i64;

        // Create input_ids tensor [batch_size, seq_len]
        let input_array = Array::from_vec(input_ids.to_vec())
            .into_shape_with_order((batch_size as usize, seq_len as usize))?;
        let input_tensor = Value::from_array(input_array)?;

        // Create attention_mask tensor [batch_size, seq_len]
        let attention_mask = vec![1i64; (batch_size * seq_len) as usize];
        let attention_array = Array::from_vec(attention_mask)
            .into_shape_with_order((batch_size as usize, seq_len as usize))?;
        let attention_tensor = Value::from_array(attention_array)?;

        // Create position_ids tensor [batch_size, seq_len]
        let position_ids: Vec<i64> = (0..seq_len).collect();
        let position_array = Array::from_vec(position_ids)
            .into_shape_with_order((batch_size as usize, seq_len as usize))?;
        let position_tensor = Value::from_array(position_array)?;

        let mut inputs: Vec<(String, Value)> = vec![
            ("input_ids".to_string(), input_tensor.into()),
            ("attention_mask".to_string(), attention_tensor.into()),
            ("position_ids".to_string(), position_tensor.into()),
        ];

        // Model expects 28 layers (0-27) based on the provided schema
        // TODO: Make dynamic
        let num_layers = 28;
        let num_heads = 8;
        let head_dim = 128;

        // Add past key-value tensors if using cache and cache is not empty
        if use_cache{
            if self.cache.is_empty(){
                // Create zero-initialized tensors with correct 4D shape
                let cache_shape = (batch_size as usize, num_heads, seq_len as usize, head_dim);
                let cache_size = batch_size as usize * num_heads * seq_len as usize * head_dim;
                debug!("Creating initial KV cache: layers={} shape={:?} size={}", num_layers, cache_shape, cache_size);
                self.cache = KVCache::new_with_size(num_layers, cache_size,  cache_shape)?;
            }

            for i in 0..num_layers.min(self.cache.len()) {
                let key_tensor = Value::from_array(self.cache.get_layer_keys(i)?)?;
                let value_tensor = Value::from_array(self.cache.get_layer_values(i)?)?;

                inputs.push((format!("past_key_values.{}.key", i), key_tensor.into()));
                inputs.push((format!("past_key_values.{}.value", i), value_tensor.into()));
            }
        }
        
        Ok(inputs)
    }

    // WIP method - many things that should not be here or might want to be later but for now they are stuck so I do not forget
    pub fn generate(
        &mut self,
        mut input_tokens: Vec<i64>,
        // current_len: i64,
        // min_tokens: i64,
        max_tokens: i64,
        // min_p: Option<f32>,
        temperature: f32,
        top_k: i64,
        top_p: f32,
        // pa1d_token_id: i64,
        // eos_token_id: i64,
        // batch_size: i64, //TODO: Not in use but want to use it
        // attention_mask: &[i64],
    ) -> Result<Vec<u32>> {
        // let starting_idx = 0;
        let mut generated_tokens: Vec<u32> = Vec::new();

        for i in 0..max_tokens {
            debug!("Generating token {}/{}", i + 1, max_tokens);

            // Run inference with current tokens
            let current_input = if i == 0 || self.cache.is_empty() {
                // First iteration: use full input to populate cache
                input_tokens.clone()
            } else {
                // Subsequent iterations: only use the last token (thanks to KV cache)
                vec![*input_tokens.last().unwrap()]
            };

            let token = self.run_inference(&current_input, temperature, top_k, top_p)?;
            if token.is_none() {
                break;
            }
            let token = token.unwrap();
            generated_tokens.push(token as u32);
            input_tokens.push(token);

            // Simple stopping condition (you'd want EOS token detection in practice)
            // TODO: true eos_token logic
            if self.end_of_sequence(token) {
                break;
            }
        }
        Ok(generated_tokens)
    }

    fn run_inference(
        &mut self,
        input_ids: &[i64],
        temperature: f32,
        top_k: i64,
        top_p: f32,
    ) -> Result<Option<i64>> {
        let inputs = self
            .create_input_tensors(input_ids, true)
            .map_err(|e| InferenceError::InputGenerationError(e.to_string()))?;
        // Convert to the format expected by ort
        let input_values: Vec<(&str, &Value)> = inputs
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();

        debug_print_inputs(&inputs);
        debug!("Using KV Cache");
        debug!("Cache Status: {} layers", self.cache.len());

        let outputs = self.session.run(input_values)?;
        debug_print_outputs(&outputs);
        

        let logits = extract_logits::<f32>(&outputs)?;
        let tokens = process_logits(logits, temperature, top_p)?;

        if tokens.is_empty() {
            return Ok(None);
        }

        // Do random sampling BEFORE updating cache to avoid borrow conflicts
        let selected_token = {
            let rng = self.rng.lock().unwrap();
            let selected = top_k_sampling(tokens, top_k, rng);
            selected.0 as i64
        }; // rng is dropped here, releasing the borrow

        // Now we can safely update cache with &mut self
        if true  { // use_cache
            // Update cache with present key-values from outputs
            // We need to extract the cache data before returning
            let mut new_keys = Vec::new();
            let mut new_values = Vec::new();

            for (output_name, output_value) in outputs.iter() {
                if output_name.contains("present") && output_name.contains("key") {
                    if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
                        let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                        if let Ok(tensor) =
                            Array::from_vec(data.to_vec()).into_shape_with_order(shape_vec)
                        {
                            new_keys.push(tensor);
                        }
                    }
                } else if output_name.contains("present")
                    && output_name.contains("value")
                    && let Ok((shape, data)) = output_value.try_extract_tensor::<f32>()
                {
                    let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                    if let Ok(tensor) = Array::from_vec(data.to_vec()).into_shape_with_order(shape_vec)
                    {
                        new_values.push(tensor);
                    }
                }
            }

            // Update cache if we found matching key-value pairs
            if !new_keys.is_empty() && new_keys.len() == new_values.len() {
                for l in 0..new_keys.len() {
                    self.cache
                        .update(l, new_keys[l].clone(), new_values[l].clone())
                        .map_err(|e| InferenceError::CacheError(e.to_string()))?;
                }

                debug!("Updated KV cache with {} layer pairs", self.cache.len());
            }
        }

        Ok(Some(selected_token))
    }

    fn end_of_sequence(&self, token: i64) -> bool {
        if let Some(eos_token) = &self.generate_cfg.eos_token_id {
            match eos_token {
                JSONValue::Array(data) => data.iter().any(|v| v.as_i64() == Some(token)),
                JSONValue::Number(val) => val.as_i64() == Some(token),
                _ => false,
            }
        } else {
            false
        }
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

    top_p_filtering(probs, top_p)
}

fn top_p_filtering(mut probs: Vec<(usize, f32)>, top_p: f32) -> Result<Vec<(usize, f32)>> {
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

fn top_k_sampling(
    probs: Vec<(usize, f32)>,
    top_k: i64,
    mut rng: MutexGuard<'_, StdRng>,
) -> (usize, f32) {
    let top_k_size = top_k.min(probs.len() as i64);
    let random_index = rng.random_range(0..top_k_size as usize);
    probs[random_index]
}

fn extract_logits<T: Into<f32> + PrimitiveTensorElementType + Copy>(
    outputs: &SessionOutputs,
) -> Result<Vec<(usize, f32)>> {
    // We grab the first output tensor from the resutls as that is 'typically' where that is stored.
    // This may need improvement later
    let (shape, probabilities) = match outputs["logits"].try_extract_tensor::<T>() {
        Ok((dim, probabilities)) => (dim, probabilities),
        Err(e) => {
            return Err(InferenceError::OutputGenerationError(format!(
                "error extracting tensor: {}",
                e
            )));
        }
    };
    // Process the output
    Ok(logits(shape.to_vec(), probabilities.to_vec()))
}

fn logits<T: Into<f32> + Copy>(shape: Vec<i64>, probabilities: Vec<T>) -> Vec<(usize, f32)> {
    // The output tensor will have shape [B, _, S, V]
    // We want only the probabilities for the last token in this sequence, which will be the next most likely token
    // according to the model
    // Output shape = [B, _, S, V]:
    //    B = 1 batch
    //    _ = 1 dummy dimension
    //    S = sequence length
    //    V = vocab size
    // That means at each time step, you get a vector of length V representing the logits/probabilities of every possible next token.
    // You don’t care about the probabilities for earlier positions (you already generated those).
    //
    // So you slice into the last V chunk:
    // The output tensor will have shape [B, _, S, V]
    // We want only the probabilities for the last token in this sequence, which will be the next most likely token
    // according to the model
    let size = shape.len();
    let (seq_len, vocab_size) = (shape[size - 2] as usize, shape[size - 1] as usize);
    probabilities[(seq_len - 1) * vocab_size..]
        .iter()
        .enumerate()
        .map(|(i, prob)| (i, (*prob).into()))
        .collect()
}

fn debug_print_inputs(inputs: &Vec<(String, Value)>){
    debug!("Input tensors:");
    for (name, value) in inputs {
        if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
            debug!(
                "  {} [i64]: shape={:?}, data={:?}",
                name,
                shape,
                if data.len() <= 20 {
                    format!("{:?}", data)
                } else {
                    format!("{:?}...", &data[..20])
                }
            );
        } else if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            debug!(
                "  {} [f32]: shape={:?}, data={:?}",
                name,
                shape,
                if data.len() <= 20 {
                    format!("{:?}", data)
                } else {
                    format!("{:?}...", &data[..20])
                }
            );
        } else {
            debug!("  {}: <unknown tensor type>", name);
        }
    }
}

fn debug_print_outputs(outputs: &SessionOutputs){
    debug!("Output tensors:");
    for (name, value) in outputs.iter() {
        if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
            debug!(
                "  {} [f32]: shape={:?}, data_len={}",
                name,
                shape,
                data.len()
            );
            if name.contains("logit") || name == "output_0" {
                let preview_len = 10.min(data.len());
                debug!(
                    "    first {} values: {:?}",
                    preview_len,
                    &data[..preview_len]
                );
                let last_preview_len = 10.min(data.len());
                if data.len() > preview_len {
                    debug!(
                        "    last {} values: {:?}",
                        last_preview_len,
                        &data[data.len() - last_preview_len..]
                    );
                }
            }
        } else if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
            debug!(
                "  {} [i64]: shape={:?}, data_len={}",
                name,
                shape,
                data.len()
            );
        } else {
            debug!("  {}: <unknown tensor type>", name);
        }
    }
}