// Optimized KV Cache Implementation
// Focuses on in-place operations and memory efficiency to avoid performance degradation

use anyhow::Error;
use std::mem;

/// A memory-efficient cache layer that uses in-place operations and pre-allocated buffers
/// to avoid expensive concatenations and cloning during text generation.
pub struct OptimizedKVCacheLayer {
    // Pre-allocated buffers that grow strategically
    keys_buffer: Vec<f32>,
    values_buffer: Vec<f32>,
    
    // Cache dimensions
    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
    
    // Current state
    current_seq_len: usize,
    max_seq_len: usize,
}

impl OptimizedKVCacheLayer {
    /// Create a new cache layer with pre-allocated capacity
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let total_capacity = batch_size * num_heads * max_seq_len * head_dim;
        
        Self {
            keys_buffer: vec![0.0f32; total_capacity],
            values_buffer: vec![0.0f32; total_capacity],
            batch_size,
            num_heads,
            head_dim,
            current_seq_len: 0,
            max_seq_len,
        }
    }

    /// Update cache with new key-value states using in-place operations
    /// This avoids expensive concatenations by directly writing to the buffer
    pub fn append_inplace(
        &mut self,
        new_keys: &[f32],
        new_values: &[f32],
        new_seq_len: usize,
    ) -> Result<(), Error> {
        if self.current_seq_len + new_seq_len > self.max_seq_len {
            return Err(anyhow::anyhow!(
                "Cache overflow: {} + {} > {}",
                self.current_seq_len,
                new_seq_len,
                self.max_seq_len
            ));
        }

        let elements_per_token = self.batch_size * self.num_heads * self.head_dim;
        let new_elements = new_seq_len * elements_per_token;

        // Validate input sizes
        if new_keys.len() != new_elements || new_values.len() != new_elements {
            return Err(anyhow::anyhow!(
                "Invalid input size: expected {}, got keys={}, values={}",
                new_elements,
                new_keys.len(),
                new_values.len()
            ));
        }

        // Calculate insertion offset
        let insert_offset = self.current_seq_len * elements_per_token;

        // In-place copy - no allocations!
        self.keys_buffer[insert_offset..insert_offset + new_elements]
            .copy_from_slice(new_keys);
        self.values_buffer[insert_offset..insert_offset + new_elements]
            .copy_from_slice(new_values);

        self.current_seq_len += new_seq_len;
        Ok(())
    }

    /// Get current cache as slices (no cloning)
    pub fn get_current_keys(&self) -> &[f32] {
        let current_elements = self.current_seq_len * self.batch_size * self.num_heads * self.head_dim;
        &self.keys_buffer[..current_elements]
    }

    pub fn get_current_values(&self) -> &[f32] {
        let current_elements = self.current_seq_len * self.batch_size * self.num_heads * self.head_dim;
        &self.values_buffer[..current_elements]
    }

    /// Get current cache shape
    pub fn get_shape(&self) -> (usize, usize, usize, usize) {
        (self.batch_size, self.num_heads, self.current_seq_len, self.head_dim)
    }

    /// Reset cache without deallocating
    pub fn reset(&mut self) {
        self.current_seq_len = 0;
        // No need to zero out the buffer - we track valid length
    }

    /// Get current sequence length
    pub fn seq_length(&self) -> usize {
        self.current_seq_len
    }

    /// Calculate memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        (self.keys_buffer.len() + self.values_buffer.len()) * mem::size_of::<f32>()
    }

    /// Calculate memory usage in MB
    pub fn size_mb(&self) -> f64 {
        self.size_bytes() as f64 / (1024.0 * 1024.0)
    }
}

/// Optimized multi-layer KV cache with memory monitoring
pub struct OptimizedKVCache {
    layers: Vec<OptimizedKVCacheLayer>,
}

impl OptimizedKVCache {
    /// Create new cache with specified dimensions
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| OptimizedKVCacheLayer::new(batch_size, num_heads, head_dim, max_seq_len))
            .collect();

        Self { layers }
    }

    /// Update a specific layer with new key-value states
    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        new_keys: &[f32],
        new_values: &[f32],
        new_seq_len: usize,
    ) -> Result<(), Error> {
        if layer_idx >= self.layers.len() {
            return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
        }

        self.layers[layer_idx].append_inplace(new_keys, new_values, new_seq_len)
    }

    /// Get layer keys as slice (no cloning)
    pub fn get_layer_keys(&self, layer_idx: usize) -> Result<&[f32], Error> {
        if layer_idx >= self.layers.len() {
            return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
        }
        Ok(self.layers[layer_idx].get_current_keys())
    }

    /// Get layer values as slice (no cloning)
    pub fn get_layer_values(&self, layer_idx: usize) -> Result<&[f32], Error> {
        if layer_idx >= self.layers.len() {
            return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
        }
        Ok(self.layers[layer_idx].get_current_values())
    }

    /// Get layer shape
    pub fn get_layer_shape(&self, layer_idx: usize) -> Result<(usize, usize, usize, usize), Error> {
        if layer_idx >= self.layers.len() {
            return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
        }
        Ok(self.layers[layer_idx].get_shape())
    }

    /// Reset all layers
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Check if cache is empty (no tokens cached)
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(|layer| layer.seq_length() == 0)
    }

    /// Get current sequence length for a layer
    pub fn seq_length(&self, layer_idx: usize) -> Result<usize, Error> {
        if layer_idx >= self.layers.len() {
            return Err(anyhow::anyhow!("Layer index {} out of bounds", layer_idx));
        }
        Ok(self.layers[layer_idx].seq_length())
    }

    /// Calculate total memory usage in bytes
    pub fn total_size_bytes(&self) -> usize {
        self.layers.iter().map(|layer| layer.size_bytes()).sum()
    }

    /// Calculate total memory usage in MB
    pub fn total_size_mb(&self) -> f64 {
        self.total_size_bytes() as f64 / (1024.0 * 1024.0)
    }

    /// Get memory breakdown per layer
    pub fn memory_breakdown(&self) -> Vec<(usize, f64, usize)> {
        self.layers
            .iter()
            .enumerate()
            .map(|(idx, layer)| {
                (idx, layer.size_mb(), layer.seq_length())
            })
            .collect()
    }

    /// Print detailed memory report
    pub fn print_memory_report(&self) {
        println!("=== KV Cache Memory Report ===");
        println!("Total layers: {}", self.num_layers());
        println!("Total memory: {:.2} MB", self.total_size_mb());
        println!("Per-layer breakdown:");
        
        for (idx, mb, seq_len) in self.memory_breakdown() {
            println!("  Layer {}: {:.2} MB (seq_len: {})", idx, mb, seq_len);
        }
        
        println!("==============================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_cache_creation() {
        let cache = OptimizedKVCache::new(4, 1, 8, 128, 100);
        assert_eq!(cache.num_layers(), 4);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_in_place_updates() {
        let mut cache = OptimizedKVCache::new(2, 1, 8, 128, 100);
        
        // Create some test data
        let keys_data = vec![1.0f32; 1 * 8 * 1 * 128]; // 1 token
        let values_data = vec![2.0f32; 1 * 8 * 1 * 128]; // 1 token
        
        // Update layer 0
        cache.update_layer(0, &keys_data, &values_data, 1).unwrap();
        
        assert_eq!(cache.seq_length(0).unwrap(), 1);
        assert!(!cache.is_empty());
        
        // Verify data integrity
        let retrieved_keys = cache.get_layer_keys(0).unwrap();
        assert_eq!(retrieved_keys.len(), keys_data.len());
        assert_eq!(retrieved_keys[0], 1.0f32);
    }

    #[test]
    fn test_memory_reporting() {
        let cache = OptimizedKVCache::new(2, 1, 8, 128, 100);
        
        // Should have predictable memory usage
        let expected_bytes_per_layer = 1 * 8 * 100 * 128 * 4 * 2; // keys + values
        let expected_total = expected_bytes_per_layer * 2; // 2 layers
        
        assert_eq!(cache.total_size_bytes(), expected_total);
    }
}
