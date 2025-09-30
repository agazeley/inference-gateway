use crate::inference::errors::{InferenceError, Result};
use ort::{tensor::TensorElementType, value::Value};

/// Create a tensor of the specified type, shape, and data
///
/// # Arguments
/// * `element_type` - The ONNX tensor element type (Int64 or Float32)
/// * `shape` - The tensor shape as a slice of dimensions
/// * `data` - The tensor data as a generic type that can be converted
pub fn create_tensor<T, S>(element_type: TensorElementType, shape: &[S], data: T) -> Result<Value>
where
    S: Into<i64> + Copy,
    T: TensorData,
{
    let shape: Vec<i64> = shape.iter().map(|&s| s.into()).collect();
    if shape.contains(&-1) {
        return Err(InferenceError::InputGenerationError(format!(
            "Invalid shape '{:?}': negative dimensions not allowed",
            shape
        )));
    }

    // Dispatch to appropriate creation function based on element type
    match element_type {
        TensorElementType::Float32 => {
            let typed_data = data.to_f32_vec()?;
            create_typed_tensor(&shape, typed_data)
        }
        TensorElementType::Float64 => {
            let typed_data = data.to_f64_vec()?;
            create_typed_tensor(&shape, typed_data)
        }
        TensorElementType::Int8 => {
            let typed_data = data.to_i8_vec()?;
            create_typed_tensor(&shape, typed_data)
        }
        TensorElementType::Int16 => {
            let typed_data = data.to_i16_vec()?;
            create_typed_tensor(&shape, typed_data)
        }
        TensorElementType::Int32 => {
            let typed_data = data.to_i32_vec()?;
            create_typed_tensor(&shape, typed_data)
        }
        TensorElementType::Int64 => {
            let typed_data = data.to_i64_vec()?;
            create_typed_tensor(&shape, typed_data)
        }
        _ => Err(InferenceError::UnsupportedDataType(format!(
            "Tensor type {:?} not supported. Only Int64 and Float32 are supported.",
            element_type
        ))),
    }
}

// Private helper methods
fn create_typed_tensor<T>(shape: &[i64], data: Vec<T>) -> Result<Value>
where
    T: ort::tensor::PrimitiveTensorElementType + std::fmt::Debug + Clone + Send + Sync + 'static,
{
    Value::from_array((shape, data))
        .map_err(|e| {
            InferenceError::InputGenerationError(format!("Failed to create tensor: {}", e))
        })
        .map(|v| v.into())
}

/// Trait for data types that can be converted to tensor data
pub trait TensorData {
    fn to_i8_vec(&self) -> Result<Vec<i8>>;
    fn to_i16_vec(&self) -> Result<Vec<i16>>;
    fn to_i32_vec(&self) -> Result<Vec<i32>>;
    fn to_i64_vec(&self) -> Result<Vec<i64>>;
    fn to_f32_vec(&self) -> Result<Vec<f32>>;
    fn to_f64_vec(&self) -> Result<Vec<f64>>;
}

// Macro to implement TensorData for multiple types
macro_rules! impl_tensor_data {
    ($type:ty) => {
        impl TensorData for $type {
            fn to_i8_vec(&self) -> Result<Vec<i8>> {
                Ok(self.iter().map(|&x| x as i8).collect())
            }

            fn to_i16_vec(&self) -> Result<Vec<i16>> {
                Ok(self.iter().map(|&x| x as i16).collect())
            }

            fn to_i32_vec(&self) -> Result<Vec<i32>> {
                Ok(self.iter().map(|&x| x as i32).collect())
            }

            fn to_i64_vec(&self) -> Result<Vec<i64>> {
                Ok(self.iter().map(|&x| x as i64).collect())
            }

            fn to_f32_vec(&self) -> Result<Vec<f32>> {
                Ok(self.iter().map(|&x| x as f32).collect())
            }

            fn to_f64_vec(&self) -> Result<Vec<f64>> {
                Ok(self.iter().map(|&x| x as f64).collect())
            }
        }
    };
}

// Use the macro to implement TensorData for Vec and slice types
impl_tensor_data!(Vec<i8>);
impl_tensor_data!(Vec<i16>);
impl_tensor_data!(Vec<i32>);
impl_tensor_data!(Vec<i64>);
impl_tensor_data!(Vec<f32>);
impl_tensor_data!(Vec<f64>);
impl_tensor_data!(&[i8]);
impl_tensor_data!(&[i16]);
impl_tensor_data!(&[i32]);
impl_tensor_data!(&[i64]);
impl_tensor_data!(&[f32]);
impl_tensor_data!(&[f64]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_int64_tensor() -> Result<()> {
        let data = vec![1i64, 2, 3, 4, 5];
        let tensor = create_tensor(TensorElementType::Int64, &[1i64, 5], data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }

    #[test]
    fn test_create_float32_tensor() -> Result<()> {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = create_tensor(TensorElementType::Float32, &[2, 2], data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }

    #[test]
    fn test_type_conversion() -> Result<()> {
        let int_data = vec![1i64, 2, 3, 4];
        let tensor = create_tensor(TensorElementType::Float32, &[4], int_data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }

    #[test]
    fn test_unsupported_type() -> Result<()> {
        let data = vec![1i64, 2, 3, 4];
        let result = create_tensor(TensorElementType::Bool, &[4], data);
        assert!(result.is_err());
        if let Err(InferenceError::UnsupportedDataType(msg)) = result {
            assert!(msg.contains("not supported"));
        }
        Ok(())
    }

    #[test]
    fn test_slice_input() -> Result<()> {
        let data: &[i64] = &[1, 2, 3, 4];
        let tensor = create_tensor(TensorElementType::Int64, &[4i64], data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }

    #[test]
    fn test_create_int8_tensor() -> Result<()> {
        let data = vec![1i8, -2, 3, 127, -128];
        let tensor = create_tensor(TensorElementType::Int8, &[5], data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }

    #[test]
    fn test_create_int16_tensor() -> Result<()> {
        let data = vec![1i16, -2, 32767, -32768];
        let tensor = create_tensor(TensorElementType::Int16, &[4], data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }

    #[test]
    fn test_create_int32_tensor() -> Result<()> {
        let data = vec![1i32, -2, 2147483647, -2147483648];
        let tensor = create_tensor(TensorElementType::Int32, &[4], data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }

    #[test]
    fn test_create_float64_tensor() -> Result<()> {
        let data = vec![1.0f64, -2.5, 3.14, 0.0];
        let tensor = create_tensor(TensorElementType::Float64, &[4], data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }

    #[test]
    fn test_create_tensor_with_negative_shape() {
        let data = vec![1i64, 2, 3, 4];
        let result = create_tensor(TensorElementType::Int64, &[-1, 4], data);
        assert!(result.is_err());
        if let Err(InferenceError::InputGenerationError(msg)) = result {
            assert!(msg.contains("negative dimensions not allowed"));
        }
    }

    #[test]
    fn test_create_tensor_with_empty_data() -> Result<()> {
        let data: Vec<i32> = vec![];
        let tensor = create_tensor(TensorElementType::Int32, &[0], data);
        assert!(tensor.is_err());
        Ok(())
    }

    #[test]
    fn test_create_tensor_with_slice_f32() -> Result<()> {
        let data: &[f32] = &[1.0, 2.0, 3.0];
        let tensor = create_tensor(TensorElementType::Float32, &[3], data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }

    #[test]
    fn test_create_tensor_with_slice_f64() -> Result<()> {
        let data: &[f64] = &[1.0, 2.0, 3.0];
        let tensor = create_tensor(TensorElementType::Float64, &[3], data)?;
        assert!(tensor.is_tensor());
        Ok(())
    }
}
