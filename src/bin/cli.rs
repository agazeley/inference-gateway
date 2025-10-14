use anyhow::{Result, anyhow};
use clap::{Arg, Command};
use hf_hub::api::sync::Api;
use inference_gateway::inference::tokenization::{Tokenizer, TokenizerConfig};
use ndarray::{Array, IxDyn};
use ort::{
    execution_providers::{CUDAExecutionProvider, ExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{fs, path::PathBuf};

// KV Cache to store past key-value pairs for efficient autoregressive generation
#[derive(Debug, Default)]
struct KVCache {
    past_keys: Vec<Array<f32, IxDyn>>,
    past_values: Vec<Array<f32, IxDyn>>,
}

impl KVCache {
    fn new() -> Self {
        Self {
            past_keys: Vec::new(),
            past_values: Vec::new(),
        }
    }

    fn update(&mut self, layer: usize, keys: Array<f32, IxDyn>, values: Array<f32, IxDyn>) {
        if self.past_keys.len() <= layer {
            self.past_keys.resize_with(layer + 1, Default::default);
        }
        if self.past_values.len() <= layer {
            self.past_values.resize_with(layer + 1, Default::default);
        }
        self.past_keys[layer] = keys;
        self.past_values[layer] = values;
    }

    fn is_empty(&self) -> bool {
        self.past_keys.is_empty() && self.past_values.is_empty()
    }
}

// Autoregressive model with KV cache
struct AutoRegressiveModel {
    session: Session,
    cache: KVCache,
    rng: StdRng,
}

impl AutoRegressiveModel {
    fn new(model_path: &str) -> Result<Self> {
        println!("Loading model from: {}", model_path);

        // Build ONNX Runtime session
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?;

        // Register CUDA provider if available
        let cuda = CUDAExecutionProvider::default();
        if cuda.is_available().unwrap_or(false) {
            if let Err(e) = cuda.register(&mut builder) {
                eprintln!("Warning: Failed to register CUDA provider: {}", e);
            } else {
                println!("CUDA provider registered for GPU acceleration");
            }
        }

        let session = if PathBuf::from(model_path).exists() {
            // Local file
            builder.commit_from_file(model_path)?
        } else {
            // Try to download from HuggingFace
            let api = Api::new()?;
            let repo = api.model(model_path.to_string());
            let local_path = repo
                .get("onnx/model_int8.onnx")
                .map_err(|e| anyhow!("Failed to download model from HF: {}", e))?;
            builder.commit_from_file(local_path)?
        };

        println!("Model loaded successfully");

        Ok(Self {
            session,
            cache: KVCache::new(),
            rng: StdRng::seed_from_u64(42),
        })
    }

    fn create_input_tensors(
        &mut self,
        input_ids: &[i64],
        use_cache: bool,
    ) -> Result<Vec<(String, Value)>> {
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

        let mut inputs = vec![
            ("input_ids".to_string(), input_tensor.into()),
            ("attention_mask".to_string(), attention_tensor.into()),
            ("position_ids".to_string(), position_tensor.into()),
        ];

        // Model expects 28 layers (0-27) based on the provided schema
        let num_layers = 28;
        let num_heads = 8;
        let head_dim = 128;

        // Add past key-value tensors if using cache and cache is not empty
        if use_cache && !self.cache.is_empty() {
            for i in 0..num_layers.min(self.cache.past_keys.len()) {
                let key_tensor = Value::from_array(self.cache.past_keys[i].clone())?;
                let value_tensor = Value::from_array(self.cache.past_values[i].clone())?;

                inputs.push((format!("past_key_values.{}.key", i), key_tensor.into()));
                inputs.push((format!("past_key_values.{}.value", i), value_tensor.into()));
            }
        } else {
            // Create initial KV cache tensors with proper 4D shape
            // Shape: [batch_size, num_heads, seq_len, head_dim]
            println!("Creating initial KV cache for {} layers", num_layers);

            for i in 0..num_layers {
                // Create zero-initialized tensors with correct 4D shape
                let cache_shape = (batch_size as usize, num_heads, seq_len as usize, head_dim);
                let cache_size = batch_size as usize * num_heads * seq_len as usize * head_dim;

                let key_data = vec![0.0f32; cache_size];
                let value_data = vec![0.0f32; cache_size];

                let key_array = Array::from_vec(key_data).into_shape_with_order(cache_shape)?;
                let value_array = Array::from_vec(value_data).into_shape_with_order(cache_shape)?;

                let key_tensor = Value::from_array(key_array.clone())?;
                let value_tensor = Value::from_array(value_array.clone())?;

                inputs.push((format!("past_key_values.{}.key", i), key_tensor.into()));
                inputs.push((format!("past_key_values.{}.value", i), value_tensor.into()));

                // Store in cache for future use
                self.cache
                    .update(i, key_array.into_dyn(), value_array.into_dyn());
            }
        }

        Ok(inputs)
    }

    fn run_inference(&mut self, input_ids: &[i64]) -> Result<Vec<f32>> {
        let use_cache = !self.cache.is_empty();
        let inputs = self.create_input_tensors(input_ids, use_cache)?;

        // Print input information
        println!("\n=== INFERENCE RUN DEBUG INFO ===");
        println!("MODEL INPUTS:");
        for input in &self.session.inputs {
            println!(
                "  {} [{:?}] shape={:?}",
                input.name,
                input.input_type.tensor_type().unwrap(),
                input.input_type.tensor_shape().unwrap()
            )
        }

        println!("MODEL OUTPUTS:");
        for output in &self.session.outputs {
            println!(
                "  {} [{:?}] shape={:?}",
                output.name,
                output.output_type.tensor_type().unwrap(),
                output.output_type.tensor_shape().unwrap()
            )
        }

        println!("\nINPUT TENSORS:");
        for (name, value) in &inputs {
            if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                println!(
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
                println!(
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
                println!("  {}: <unknown tensor type>", name);
            }
        }
        println!("🎯 Using KV Cache: {}", use_cache);
        println!(
            "Cache Status: {} key layers, {} value layers",
            self.cache.past_keys.len(),
            self.cache.past_values.len()
        );

        // Convert to the format expected by ort
        let input_values: Vec<(&str, &Value)> = inputs
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();

        let outputs = self.session.run(input_values)?;

        // Print output information
        println!("\nOUTPUT TENSORS:");
        for (name, value) in outputs.iter() {
            if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                println!(
                    "  {} [f32]: shape={:?}, data_len={}",
                    name,
                    shape,
                    data.len()
                );
                if name.contains("logit") || name == "output_0" {
                    let preview_len = 10.min(data.len());
                    println!(
                        "    first {} values: {:?}",
                        preview_len,
                        &data[..preview_len]
                    );
                    let last_preview_len = 10.min(data.len());
                    if data.len() > preview_len {
                        println!(
                            "    last {} values: {:?}",
                            last_preview_len,
                            &data[data.len() - last_preview_len..]
                        );
                    }
                }
            } else if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
                println!(
                    "  {} [i64]: shape={:?}, data_len={}",
                    name,
                    shape,
                    data.len()
                );
            } else {
                println!("  {}: <unknown tensor type>", name);
            }
        }
        println!("=== END DEBUG INFO ===\n");

        // Extract logits (typically the first output named "logits" or similar)
        let logits_value = outputs.get("logits").unwrap_or(&outputs[0]);
        let (shape, data) = logits_value.try_extract_tensor::<f32>()?;

        // Get the last token's logits - properly extract from tensor shape
        let vocab_size = shape[shape.len() - 1] as usize; // Last dimension is vocab size
        let last_token_logits = if data.len() >= vocab_size {
            data[data.len() - vocab_size..].to_vec()
        } else {
            data.to_vec()
        };

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
                    .update(l, new_keys[l].clone(), new_values[l].clone());
            }

            println!(
                "Updated KV cache with {} layer pairs",
                self.cache.past_keys.len()
            );
        }

        Ok(last_token_logits)
    }

    fn sample_token(&mut self, logits: &[f32], temperature: f32, top_k: usize) -> usize {
        if logits.is_empty() {
            return 0;
        }

        // Apply temperature scaling
        let scaled_logits: Vec<f32> = logits.iter().map(|&logit| logit / temperature).collect();

        // Find max for numerical stability
        let max_logit = scaled_logits
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Convert to probabilities
        let exp_logits: Vec<f32> = scaled_logits
            .iter()
            .map(|&logit| (logit - max_logit).exp())
            .collect();

        let sum: f32 = exp_logits.iter().sum();
        let mut probs: Vec<(usize, f32)> = exp_logits
            .iter()
            .enumerate()
            .map(|(i, &prob)| (i, prob / sum))
            .collect();

        // Sort by probability (descending)
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k sampling
        let k = top_k.min(probs.len());
        probs.truncate(k);

        // Renormalize
        let sum_k: f32 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum_k;
        }

        // Sample from the distribution
        let random_val: f32 = self.rng.random();
        let mut cumulative = 0.0;

        for (token_id, prob) in &probs {
            cumulative += prob;
            if random_val <= cumulative {
                return *token_id;
            }
        }

        // Fallback to most likely token
        probs.first().map(|(id, _)| *id).unwrap_or(0)
    }

    fn generate(
        &mut self,
        input_text: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        tokenizer: &Tokenizer,
    ) -> Result<String> {
        println!("Tokenizing input text...");
        let mut input_tokens: Vec<i64> = tokenizer
            .encode(input_text, false)?
            .get_ids()
            .iter()
            .map(|i| *i as i64)
            .collect();
        println!("Input tokens: {:?}", input_tokens);

        let mut generated_tokens = Vec::new();

        println!("Starting autoregressive generation with KV cache...");
        for i in 0..max_tokens {
            println!("Generating token {}/{}", i + 1, max_tokens);

            // Run inference with current tokens
            let current_input = if i == 0 || self.cache.is_empty() {
                // First iteration: use full input to populate cache
                input_tokens.clone()
            } else {
                // Subsequent iterations: only use the last token (thanks to KV cache)
                vec![*input_tokens.last().unwrap()]
            };

            let logits = self.run_inference(&current_input)?;

            // Sample next token
            let next_token = self.sample_token(&logits, temperature, top_k);
            generated_tokens.push(next_token as u32);
            input_tokens.push(next_token as i64);

            // Simple stopping condition (you'd want EOS token detection in practice)
            if next_token == 0 || generated_tokens.len() >= max_tokens {
                break;
            }
        }

        // Convert tokens back to text (simplified)
        let generated_text: String = tokenizer
            .decode(&generated_tokens, true)?
            .trim_start()
            .to_string();

        Ok(generated_text)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let matches = Command::new("Inference Gateway")
        .version("1.0")
        .author("Your Name <your.email@example.com>")
        .about("Runs an autoregressive ML model with a KV cache")
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .value_name("MODEL_ID")
                .help("HuggingFace model ID or local path")
                .required(true),
        )
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("TEXT")
                .help("Input text or path to text file")
                .required(true),
        )
        .arg(
            Arg::new("max_tokens")
                .long("max-tokens")
                .value_name("N")
                .help("Maximum number of tokens to generate")
                .default_value("50"),
        )
        .arg(
            Arg::new("temperature")
                .long("temperature")
                .value_name("TEMP")
                .help("Sampling temperature (0.1 to 2.0)")
                .default_value("1.0"),
        )
        .arg(
            Arg::new("top_k")
                .long("top-k")
                .value_name("K")
                .help("Top-k sampling parameter")
                .default_value("50"),
        )
        .get_matches();

    let model_id = matches.get_one::<String>("model").unwrap();
    let input_text = matches.get_one::<String>("input").unwrap();
    let max_tokens: usize = matches.get_one::<String>("max_tokens").unwrap().parse()?;
    let temperature: f32 = matches.get_one::<String>("temperature").unwrap().parse()?;
    let top_k: usize = matches.get_one::<String>("top_k").unwrap().parse()?;

    // Determine if input is a file path or direct text
    let input_content = if PathBuf::from(input_text).exists() {
        println!("Reading input from file: {}", input_text);
        fs::read_to_string(input_text)?
    } else {
        println!("Using direct input text: {}", input_text);
        input_text.clone()
    };

    // Initialize the autoregressive model
    let mut model = AutoRegressiveModel::new(model_id)?;
    let tokenizer = Tokenizer::new(
        TokenizerConfig {
            filepath: None,
            pretrained_identifier: Some(model_id.clone()),
        },
        None,
    )?;

    // Generate text
    println!("\n=== Starting Text Generation ===");
    println!("Parameters:");
    println!("  Max tokens: {}", max_tokens);
    println!("  Temperature: {}", temperature);
    println!("  Top-k: {}", top_k);
    println!();

    let generated_text =
        model.generate(&input_content, max_tokens, temperature, top_k, &tokenizer)?;

    println!("\n=== Generation Complete ===");
    println!("Input: {}", input_content);
    println!("Generated: {}", generated_text);
    println!();
    println!("✅ Autoregressive generation with KV cache completed successfully!");

    Ok(())
}
