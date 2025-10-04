// KV Cache
// Taken inspiration for the API from: https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py

use anyhow::Ok;
use ndarray::{ArrayD, Axis, concatenate};

// TODO: Make this not statically f32
// TODO: Make this more thread and memory safe using Arc + Mutex on the Arrays
// A cache layer that grows dynamically as more tokens are generated. This is the default for generative models.
// It stores the key and value states as tensors of shape `[batch_size, num_heads, seq_len, head_dim]`.
pub struct KVCacheLayer {
    keys: Option<ArrayD<f32>>,
    values: Option<ArrayD<f32>>,
}

impl KVCacheLayer {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
        }
    }

    pub fn update(
        &mut self,
        key_states: ArrayD<f32>,
        value_states: ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>), anyhow::Error> {
        // dim = -2 in PyTorch → second-to-last axis
        let axis = key_states.ndim() - 2;

        // Concatenate new keys
        self.keys = Some(match &self.keys {
            Some(existing) => concatenate(Axis(axis), &[existing.view(), key_states.view()])?,
            None => key_states,
        });

        // Concatenate new values
        self.values = Some(match &self.values {
            Some(existing) => concatenate(Axis(axis), &[existing.view(), value_states.view()])?,
            None => value_states,
        });

        // Return clones of the arrays (like Python would)
        Ok((
            self.keys.as_ref().unwrap().clone(),
            self.values.as_ref().unwrap().clone(),
        ))
    }

    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
    }

    pub fn get_seq_length(&self) -> i64 {
        match &self.keys {
            Some(keys) => {
                // Sequence length is typically the second-to-last dimension
                let seq_dim = keys.ndim() - 2;
                keys.shape()[seq_dim] as i64
            }
            None => 0,
        }
    }

    pub fn get_mask_sizes(&self, cache_position: &ArrayD<f32>) -> (i64, i64) {
        // Return the length and offset of the cache, used to generate the mask
        let kv_offset = 0;
        let query_length = cache_position.shape()[0] as i64; // cache_position.shape[0] in Python
        let kv_length = self.get_seq_length() + query_length;
        (kv_length, kv_offset)
    }

    /// Get a copy of the keys cache (like Python would return the tensor)
    pub fn get_keys(&self) -> Option<ArrayD<f32>> {
        self.keys.clone()
    }

    /// Get a copy of the values cache (like Python would return the tensor)
    pub fn get_values(&self) -> Option<ArrayD<f32>> {
        self.values.clone()
    }
}

pub struct KVCache {
    layers: Option<Vec<KVCacheLayer>>,
}

impl Default for KVCache {
    fn default() -> Self {
        Self { layers: None }
    }
}

impl KVCache {
    pub fn new(layers: Vec<KVCacheLayer>) -> Self {
        Self {
            layers: Some(layers),
        }
    }

    pub fn reset(&mut self) {
        if let Some(layers) = &mut self.layers {
            for layer in layers.iter_mut() {
                layer.reset();
            }
        }
    }

    pub fn update(
        &mut self,
        layer_idx: usize,
        key_states: ArrayD<f32>,
        value_states: ArrayD<f32>,
    ) -> Result<(ArrayD<f32>, ArrayD<f32>), anyhow::Error> {
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
        cache_position: &ArrayD<f32>,
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
    pub fn get_layer_keys(&self, layer_idx: usize) -> Result<Option<ArrayD<f32>>, anyhow::Error> {
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
    pub fn get_layer_values(&self, layer_idx: usize) -> Result<Option<ArrayD<f32>>, anyhow::Error> {
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
#[cfg(test)]
mod tests {
    use ndarray::IxDyn;

    use super::*;

    fn create_array(shape: &[usize], val: f32) -> ArrayD<f32> {
        ArrayD::from_elem(IxDyn(shape), val)
    }

    #[test]
    fn test_kvcachelayer_update_and_reset() {
        let mut layer = KVCacheLayer::new();
        let key1 = create_array(&[1, 2, 3, 4], 1.0);
        let value1 = create_array(&[1, 2, 3, 4], 2.0);

        let (keys, values) = layer.update(key1.clone(), value1.clone()).unwrap();
        assert_eq!(keys, key1);
        assert_eq!(values, value1);

        let key2 = create_array(&[1, 2, 2, 4], 3.0);
        let value2 = create_array(&[1, 2, 2, 4], 4.0);

        let (keys2, values2) = layer.update(key2.clone(), value2.clone()).unwrap();
        assert_eq!(keys2.shape()[2], 5); // 3 + 2 along axis 2
        assert_eq!(values2.shape()[2], 5);

        layer.reset();
        assert!(layer.get_keys().is_none());
        assert!(layer.get_values().is_none());
    }

    #[test]
    fn test_kvcachelayer_get_seq_length() {
        let mut layer = KVCacheLayer::new();
        assert_eq!(layer.get_seq_length(), 0);

        let key = create_array(&[1, 2, 3, 4], 1.0);
        let value = create_array(&[1, 2, 3, 4], 2.0);
        layer.update(key, value).unwrap();
        assert_eq!(layer.get_seq_length(), 3);
    }

    #[test]
    fn test_kvcachelayer_get_mask_sizes() {
        let mut layer = KVCacheLayer::new();
        let key = create_array(&[1, 2, 3, 4], 1.0);
        let value = create_array(&[1, 2, 3, 4], 2.0);
        layer.update(key, value).unwrap();

        let cache_position = create_array(&[5], 0.0);
        let (kv_length, kv_offset) = layer.get_mask_sizes(&cache_position);
        assert_eq!(kv_length, 8); // 3 + 5
        assert_eq!(kv_offset, 0);
    }

    #[test]
    fn test_kvcache_update_and_getters() {
        let mut cache = KVCache::new(vec![KVCacheLayer::new(), KVCacheLayer::new()]);
        let key = create_array(&[1, 2, 3, 4], 1.0);
        let value = create_array(&[1, 2, 3, 4], 2.0);

        let (keys, values) = cache.update(0, key.clone(), value.clone()).unwrap();
        assert_eq!(keys, key);
        assert_eq!(values, value);

        assert_eq!(cache.get_seq_length(0).unwrap(), 3);

        let cache_position = create_array(&[2], 0.0);
        let (kv_length, kv_offset) = cache.get_mask_sizes(&cache_position, 0).unwrap();
        assert_eq!(kv_length, 5); // 3 + 2
        assert_eq!(kv_offset, 0);

        let layer_keys = cache.get_layer_keys(0).unwrap().unwrap();
        assert_eq!(layer_keys, key);

        let layer_values = cache.get_layer_values(0).unwrap().unwrap();
        assert_eq!(layer_values, value);
    }

    #[test]
    fn test_kvcache_reset() {
        let mut cache = KVCache::new(vec![KVCacheLayer::new()]);
        let key = create_array(&[1, 2, 3, 4], 1.0);
        let value = create_array(&[1, 2, 3, 4], 2.0);
        cache.update(0, key, value).unwrap();

        cache.reset();
        assert!(cache.get_layer_keys(0).unwrap().is_none());
        assert!(cache.get_layer_values(0).unwrap().is_none());
    }

    #[test]
    fn test_kvcache_out_of_bounds() {
        let mut cache = KVCache::new(vec![KVCacheLayer::new()]);
        let key = create_array(&[1, 2, 3, 4], 1.0);
        let value = create_array(&[1, 2, 3, 4], 2.0);

        assert!(cache.update(1, key.clone(), value.clone()).is_err());
        assert!(cache.get_seq_length(1).is_err());
        let cache_position = create_array(&[2], 0.0);
        assert!(cache.get_mask_sizes(&cache_position, 1).is_err());
        assert!(cache.get_layer_keys(1).is_err());
        assert!(cache.get_layer_values(1).is_err());
    }

    #[test]
    fn test_kvcache_no_layers_initialized() {
        let mut cache = KVCache::default();
        let key = create_array(&[1, 2, 3, 4], 1.0);
        let value = create_array(&[1, 2, 3, 4], 2.0);

        assert!(cache.update(0, key.clone(), value.clone()).is_err());
        assert!(cache.get_seq_length(0).is_err());
        let cache_position = create_array(&[2], 0.0);
        assert!(cache.get_mask_sizes(&cache_position, 0).is_err());
        assert!(cache.get_layer_keys(0).is_err());
        assert!(cache.get_layer_values(0).is_err());
    }
}
