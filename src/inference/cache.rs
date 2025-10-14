// KV Cache
// Taken inspiration for the API from: https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py

use anyhow::Error;
use ndarray::{Array, Axis, IxDyn, ShapeArg, concatenate};

// TODO: Make this not statically f32
// TODO: Make this more thread and memory safe using Arc + Mutex on the Arrays
// A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
// It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
pub struct KVCacheLayer {
    keys: Array<f32, IxDyn>,
    values: Array<f32, IxDyn>,
}

impl KVCacheLayer {
    pub fn new(keys: Array<f32, IxDyn>, values: Array<f32, IxDyn>) -> Self {
        Self { keys, values }
    }

    pub fn default<S>(size: usize, shape: S) -> Result<Self, Error>
    where
        S: ShapeArg + Clone,
    {
        // Create separate data vectors for keys and values
        let keys_data = vec![0.0f32; size];
        let values_data = vec![0.0f32; size];
        
        // Create separate arrays
        let keys_arr = Array::from_vec(keys_data)
            .into_shape_with_order(shape.clone())?
            .into_dyn();
            
        let values_arr = Array::from_vec(values_data)
            .into_shape_with_order(shape)?
            .into_dyn();
            
        Ok(Self::new(keys_arr, values_arr))
    }

    pub fn update(
        &mut self,
        key_states: Array<f32, IxDyn>,
        value_states: Array<f32, IxDyn>,
    ) -> Result<(Array<f32, IxDyn>, Array<f32, IxDyn>), anyhow::Error> {
    
        // dim = -2 in PyTorch → second-to-last axis
        let axis = key_states.ndim() - 2; 

        // Concatenate new keys
        self.keys = concatenate(Axis(axis), &[self.keys.view(), key_states.view()])?;

        // Concatenate new values
        self.values = concatenate(Axis(axis), &[self.values.view(), value_states.view()])?;

        // Return clones of the arrays (like Python would)
        Ok((self.keys.clone(), self.values.clone()))
    }

    pub fn reset(&mut self) {
        self.keys = Array::default(self.keys.shape());
        self.values = Array::default(self.values.shape());
    }

    pub fn get_seq_length(&self) -> i64 {
        let seq_dim = self.keys.ndim() - 2;
        self.keys.shape()[seq_dim] as i64
    }

    pub fn get_mask_sizes(&self, cache_position: &Array<f32, IxDyn>) -> (i64, i64) {
        // Return the length and offset of the cache, used to generate the mask
        let kv_offset = 0;
        let query_length = cache_position.shape()[0] as i64; // cache_position.shape[0] in Python
        let kv_length = self.get_seq_length() + query_length;
        (kv_length, kv_offset)
    }

    /// Get a copy of the keys cache (like Python would return the tensor)
    pub fn get_keys(&self) -> Array<f32, IxDyn> {
        self.keys.clone()
    }

    /// Get a copy of the values cache (like Python would return the tensor)
    pub fn get_values(&self) -> Array<f32, IxDyn> {
        self.values.clone()
    }
}

#[derive(Default)]
pub struct KVCache {
    layers: Option<Vec<KVCacheLayer>>,
}

impl KVCache {
    pub fn new(layers: Vec<KVCacheLayer>) -> Self {
        Self {
            layers: Some(layers),
        }
    }

    pub fn new_with_size<S>(num_layers: usize, elements_per_layer: usize, shape: S) -> Result<Self, Error>
    where
        S: ShapeArg + Clone,
    {
        
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(KVCacheLayer::default(elements_per_layer, shape.clone())?);
        }
        Ok(Self::new(layers))
    }

    pub fn reset(&mut self) {
        if let Some(layers) = &mut self.layers {
            for layer in layers.iter_mut() {
                layer.reset();
            }
        }
    }

    pub fn len(&self) -> usize {
        match &self.layers {
            Some(layers) => layers.len(),
            None => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.layers {
            Some(layers) => layers.is_empty(),
            None => true,
        }
    }

    pub fn update(
        &mut self,
        layer_idx: usize,
        key_states: Array<f32, IxDyn>,
        value_states: Array<f32, IxDyn>,
    ) -> Result<(Array<f32, IxDyn>, Array<f32, IxDyn>), anyhow::Error> {
        match &mut self.layers {
            Some(layers) => {
                if layer_idx >= layers.len() {
                    return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
                }
                layers[layer_idx].update(key_states, value_states)
            }
            None => Err(anyhow::anyhow!("No layers initialized")),
        }
    }

    pub fn get_seq_length(&self, layer_idx: usize) -> Result<i64, anyhow::Error> {
        match &self.layers {
            Some(layers) => {
                if layer_idx >= layers.len() {
                    return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
                }
                Ok(layers[layer_idx].get_seq_length())
            }
            None => Err(anyhow::anyhow!("No layers initialized")),
        }
    }

    pub fn get_mask_sizes(
        &self,
        cache_position: &Array<f32, IxDyn>,
        layer_idx: usize,
    ) -> Result<(i64, i64), anyhow::Error> {
        match &self.layers {
            Some(layers) => {
                if layer_idx >= layers.len() {
                    return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
                }
                Ok(layers[layer_idx].get_mask_sizes(cache_position))
            }
            None => Err(anyhow::anyhow!("No layers initialized")),
        }
    }

    /// Get a copy of the keys cache for a specific layer (like Python)
    pub fn get_layer_keys(&self, layer_idx: usize) -> Result<Array<f32, IxDyn>, anyhow::Error> {
        match &self.layers {
            Some(layers) => {
                if layer_idx >= layers.len() {
                    return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
                }
                Ok(layers[layer_idx].get_keys())
            }
            None => Err(anyhow::anyhow!("No layers initialized")),
        }
    }

    /// Get a copy of the values cache for a specific layer (like Python)
    pub fn get_layer_values(&self, layer_idx: usize) -> Result<Array<f32, IxDyn>, anyhow::Error> {
        match &self.layers {
            Some(layers) => {
                if layer_idx >= layers.len() {
                    return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
                }
                Ok(layers[layer_idx].get_values())
            }
            None => Err(anyhow::anyhow!("No layers initialized")),
        }
    }
}
