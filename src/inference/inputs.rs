/// # Inference Input/Output Utilities
///
/// This module provides utilities for dynamically building input tensors and extracting output tensors
/// for ONNX-based transformer models. It adapts to model signatures, resolves dynamic shapes, and
/// facilitates the construction and extraction of tensors for inference.
///
/// ## Key Structures
///
/// - **Signature**: Represents the name, shape, and data type of a model input or output.
/// - **InputBuilder**: Dynamically builds input tensors based on model signatures and token data.
/// - **OutputBuilder**: Extracts output tensors from inference results using output signatures.
/// - **ModelOutput**: Holds the shape and probabilities/logits for a model output tensor.
///
/// ## Main Features
///
/// - **Dynamic Shape Resolution**: Handles ONNX dynamic dimensions (e.g., `-1`) and resolves them
///   based on the input token sequence length.
/// - **Input Tensor Generation**: Supports common transformer input names (`input_ids`, `attention_mask`,
///   `position_ids`, and cache tensors), creating tensors with appropriate shapes and values.
/// - **Output Extraction**: Retrieves output tensors by name or index, and extracts logits for the last
///   token in the sequence.
/// - **Error Handling**: Provides detailed error messages for unsupported shapes, missing outputs,
///   and extraction failures.
///
/// ## Notes
///
/// - Assumes ONNX model input/output types are compatible with the provided utilities.
/// - Dynamic dimensions are resolved according to common transformer conventions (batch size = 1, sequence length = token count).
/// - Error handling is robust for unsupported shapes and extraction failures.
use log::{debug, trace};
use ort::{
    session::{Input, Output, SessionOutputs},
    tensor::{PrimitiveTensorElementType, TensorElementType},
    value::Value,
};
use std::fmt::{self};

use crate::inference::{
    errors::{InferenceError, Result},
    tensors::create_tensor,
};

// Represents the input signature of a model input
#[derive(Debug, Clone)]
pub struct Signature {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: TensorElementType,
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Signature({}, shape={:?}, data_type={})",
            self.name, self.shape, self.data_type
        )
    }
}

// Dynamic input builder that adapts to model signatures to build input tensors
#[derive(Debug)]
pub struct InputBuilder {
    signatures: Vec<Signature>,
}

impl InputBuilder {
    pub fn from_session_inputs(inputs: &[Input]) -> Self {
        let signatures = inputs
            .iter()
            .map(|input| {
                Signature {
                    name: input.name.clone(),
                    shape: input.input_type.tensor_shape().unwrap().to_vec(), // danger!
                    data_type: input.input_type.tensor_type().unwrap(),       // danger!
                }
            })
            .collect();

        Self { signatures }
    }

    /// Generate input tensors based on the model's signature and token data
    pub fn build(&self, tokens: &[i64]) -> Result<Vec<(String, Value)>> {
        let mut inputs = Vec::new();

        for signature in &self.signatures {
            let tensor = self.create_tensor_for_signature(signature, tokens, None);
            if let Ok(t) = tensor {
                inputs.push((signature.name.clone(), t));
            } else {
                return Err(InferenceError::InputGenerationError(
                    tensor.err().unwrap().to_string(),
                ));
            }
        }

        inputs
            .iter()
            .for_each(|(i, v)| trace!("{} -> {:?}", i, v.shape()));

        Ok(inputs)
    }

    #[expect(clippy::needless_range_loop)] // ignore for creation of position_ids
    fn create_tensor_for_signature(
        &self,
        signature: &Signature,
        tokens: &[i64],
        _past_sequence: Option<&[i64]>,
    ) -> Result<Value> {
        // Create the tensor based on common input patterns
        match signature.name.as_str() {
            // Common transformer input names
            "input_ids" | "input" | "tokens" | "input1" => {
                // Realize dynamic (-1) dimensions based on tensor type and token data
                let shape = realize_shape(&signature.shape, tokens.len() as i64)?;
                create_tensor(signature.data_type, &shape, tokens)
            }
            "attention_mask" => {
                // Realize dynamic (-1) dimensions based on tensor type and token data
                let shape = realize_shape(&signature.shape, tokens.len() as i64)?;
                let expected_size: usize = shape.iter().map(|&d| d as usize).product();
                let attention_data = vec![1i64; expected_size];
                create_tensor(signature.data_type, &shape, attention_data)
            }
            "position_ids" => {
                // Realize dynamic (-1) dimensions based on tensor type and token data
                let shape = realize_shape(&signature.shape, tokens.len() as i64)?;
                let expected_size: usize = shape.iter().map(|&d| d as usize).product();
                let mut position_data = vec![0i64; expected_size];

                // Fill with sequential position IDs up to the sequence length
                for i in 0..tokens.len().min(expected_size) {
                    position_data[i] = i as i64;
                }
                create_tensor(signature.data_type, &shape, position_data)
            }
            // Past key / value cache tensors (float32). Example names: past_key_values.0.key
            _ if signature.name.starts_with("past_key_values.") => {
                let shape = realize_shape(&signature.shape, tokens.len() as i64)?;
                let expected_size: usize = shape.iter().map(|&d| d as usize).product();
                let data = vec![0i64; expected_size]; // This creates Vec::new() when expected_size is 0
                create_tensor(signature.data_type, &shape, data)
            }
            _ => Err(InferenceError::InputGenerationError(
                "Not a supported input name".to_string(),
            )),
        }
    }
}

/// Resolves dynamic tensor shape dimensions to concrete values based on token sequence length.
///
/// ONNX models often use dynamic dimensions (represented as -1) in their input tensor shapes
/// to accommodate variable-length sequences. This function converts those dynamic dimensions
/// to actual values based on the input token sequence length.
///
/// # Shape Resolution Rules
///
/// The function handles different tensor dimensionalities with specific rules:
///
/// - **1D tensors**: `[-1]` becomes `[token_length]`
/// - **2D tensors**: The first dynamic dimension becomes 1 (batch size),
///   the second becomes `token_length` (sequence length)
/// - **3D tensors**: Similar to 2D, with the last dynamic dimension becoming `token_length`
/// - **Higher dimensions**: First dynamic dimension becomes 1, others become 0
///
/// # Arguments
///
/// * `orig` - The original tensor shape from the model signature, may contain -1 for dynamic dimensions
/// * `token_length` - The actual length of the input token sequence
///
/// # Returns
///
/// A new vector with all dynamic dimensions (-1) replaced with concrete values
fn realize_shape(orig: &[i64], token_length: i64) -> Result<Vec<i64>> {
    match orig.len() {
        1 => Ok(vec![if orig[0] < 0 { token_length } else { orig[0] }]),
        2 => Ok(vec![
            if orig[0] < 0 { 1 } else { orig[0] },
            if orig[1] < 0 { token_length } else { orig[1] },
        ]),
        3 => Ok(vec![
            if orig[0] < 0 { 1 } else { orig[0] },
            if orig[1] < 0 { 1 } else { orig[1] },
            if orig[2] < 0 { token_length } else { orig[2] },
        ]),
        4 => Ok(orig
            .iter()
            .map(|&d| {
                if d >= 0 {
                    d
                } else {
                    1 // might use token_length? Onnx runtime requires all to be >=1
                }
            })
            .collect()),
        _ => Err(InferenceError::UnsupportedDataType(format!(
            "Shape length {:?} is not supported",
            orig.len()
        ))),
    }
}

// Dynamic input builder that adapts to model signatures to generate ModelOutput's
pub struct OutputBuilder {
    outputs: Vec<Signature>,
}

/// Builder for constructing output signatures and extracting model outputs from session results.
///
/// # Methods
///
/// - `from_session_outputs(outputs: &[Output]) -> Self`
///   Constructs an `OutputBuilder` from a slice of `Output` objects, extracting their signatures (name, shape, and data type).
///
/// - `get<T>(&self, key: &str, outputs: SessionOutputs) -> Result<ModelOutput<T>>`
///   Retrieves and extracts a tensor output of type `T` from the session outputs by key. Returns an error if the key does not exist or extraction fails.
///
/// - `get_first<T>(&self, outputs: SessionOutputs) -> Result<ModelOutput<T>>`
///   Retrieves and extracts the first tensor output of type `T` from the session outputs. Returns an error if no outputs are found.
impl OutputBuilder {
    pub fn from_session_outputs(outputs: &[Output]) -> Self {
        let signatures: Vec<Signature> = outputs
            .iter()
            .map(|o| {
                Signature {
                    name: o.name.clone(),
                    shape: o.output_type.tensor_shape().unwrap().to_vec(), // danger!
                    data_type: o.output_type.tensor_type().unwrap(),       // danger!
                }
            })
            .collect();
        debug!("Outputs");
        signatures.iter().for_each(|s| {
            debug!("{:?}", s);
        });
        Self {
            outputs: signatures,
        }
    }

    pub fn get<T: PrimitiveTensorElementType + Copy>(
        &self,
        key: &str,
        outputs: SessionOutputs,
    ) -> Result<ModelOutput<T>> {
        if !outputs.contains_key(key) {
            return Err(InferenceError::OutputGenerationError(format!(
                "output '{}' does not exis in keys {:?}",
                key,
                outputs.keys().collect::<Vec<&str>>()
            )));
        }
        let (shape, probabilities) = match outputs[key].try_extract_tensor::<T>() {
            Ok((dim, probabilities)) => (dim, probabilities),
            Err(e) => {
                return Err(InferenceError::OutputGenerationError(format!(
                    "error extracting tensor: {}",
                    e
                )));
            }
        };
        Ok(ModelOutput::<T> {
            shape: shape.to_vec(),
            probabilities: probabilities.to_vec(),
        })
    }

    pub fn get_first<T: PrimitiveTensorElementType + Copy>(
        &self,
        outputs: SessionOutputs,
    ) -> Result<ModelOutput<T>> {
        if self.outputs.is_empty() {
            return Err(InferenceError::OutputGenerationError(
                "no outputs found".to_string(),
            ));
        }
        let first_signature = &self.outputs[0];
        self.get::<T>(first_signature.name.as_str(), outputs)
    }
}

pub struct ModelOutput<T> {
    shape: Vec<i64>,
    probabilities: Vec<T>,
}

/// Returns the logits (probabilities) for the last token in the sequence from the model output.
///
/// The output tensor is expected to have the shape `[B, _, S, V]`, where:
/// - `B` is the batch size (typically 1),
/// - `_` is a dummy dimension,
/// - `S` is the sequence length,
/// - `V` is the vocabulary size.
///
/// This method extracts the logits for the last position in the sequence (`S - 1`), which represent
/// the probabilities of every possible next token according to the model. Earlier positions are ignored,
/// as their probabilities have already been generated.
///
/// # Returns
/// A `Result` containing a vector of tuples, where each tuple consists of the token index and its
/// corresponding probability as an `f32`.
///
/// # Errors
/// Returns an error if the slicing operation fails due to invalid tensor shape or indices.
impl<T: Into<f32> + Copy> ModelOutput<T> {
    pub fn logits(&self) -> Result<Vec<(usize, f32)>> {
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
        let size = self.shape.len();
        let (seq_len, vocab_size) = (self.shape[size - 2] as usize, self.shape[size - 1] as usize);
        Ok(self.probabilities[(seq_len - 1) * vocab_size..]
            .iter()
            .enumerate()
            .map(|(i, prob)| (i, (*prob).into()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realize_shape_1d_dynamic() {
        let shape = vec![-1];
        let result = realize_shape(&shape, 10).unwrap();
        assert_eq!(result, vec![10]);
    }

    #[test]
    fn test_realize_shape_1d_fixed() {
        let shape = vec![5];
        let result = realize_shape(&shape, 10).unwrap();
        assert_eq!(result, vec![5]);
    }

    #[test]
    fn test_realize_shape_2d_both_dynamic() {
        let shape = vec![-1, -1];
        let result = realize_shape(&shape, 15).unwrap();
        assert_eq!(result, vec![1, 15]);
    }

    #[test]
    fn test_realize_shape_2d_first_dynamic() {
        let shape = vec![-1, 8];
        let result = realize_shape(&shape, 12).unwrap();
        assert_eq!(result, vec![1, 8]);
    }

    #[test]
    fn test_realize_shape_2d_second_dynamic() {
        let shape = vec![3, -1];
        let result = realize_shape(&shape, 20).unwrap();
        assert_eq!(result, vec![3, 20]);
    }

    #[test]
    fn test_realize_shape_2d_both_fixed() {
        let shape = vec![4, 6];
        let result = realize_shape(&shape, 25).unwrap();
        assert_eq!(result, vec![4, 6]);
    }

    #[test]
    fn test_realize_shape_3d_all_dynamic() {
        let shape = vec![-1, -1, -1];
        let result = realize_shape(&shape, 30).unwrap();
        assert_eq!(result, vec![1, 1, 30]);
    }

    #[test]
    fn test_realize_shape_3d_mixed() {
        let shape = vec![2, -1, 7];
        let result = realize_shape(&shape, 18).unwrap();
        assert_eq!(result, vec![2, 1, 7]);
    }

    #[test]
    fn test_realize_shape_4d_general_case() {
        let shape = vec![-1, 4, -1, 8];
        let result = realize_shape(&shape, 50).unwrap();
        assert_eq!(result, vec![1, 4, 1, 8]);
    }

    #[test]
    fn test_realize_shape_empty() {
        let shape = vec![];
        assert!(realize_shape(&shape, 10).is_err());
    }

    #[test]
    fn test_realize_shape_zero_token_length() {
        let shape = vec![-1, -1];
        let result = realize_shape(&shape, 0).unwrap();
        assert_eq!(result, vec![1, 0]);
    }

    #[test]
    fn test_realize_shape_large_dimensions() {
        let shape = vec![-1, -1, -1, -1, -1];
        assert!(realize_shape(&shape, 100).is_err());
        // assert_eq!(result, vec![1, 100, 100, 100, 100]);
    }
}
