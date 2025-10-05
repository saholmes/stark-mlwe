use ark_ff::PrimeField;
use blake3::Hasher;
use ark_pallas::Fr as F;
use serde::{Deserialize, Serialize};

/// Map arbitrary bytes to a field element by reducing mod p (little-endian).
pub fn fr_from_le_bytes_mod_p(bytes: &[u8]) -> F {
    // Use a 512-bit buffer to avoid bias from short inputs
    let mut wide = [0u8; 64];
    let len = bytes.len().min(64);
    wide[..len].copy_from_slice(&bytes[..len]);
    F::from_le_bytes_mod_order(&wide)
}

/// Hash(tag || data) with BLAKE3, then map to Fr via reduce-mod-p.
pub fn fr_from_hash(tag: &str, data: &[u8]) -> F {
    let mut h = Hasher::new();
    h.update(tag.as_bytes());
    h.update(data);
    let out = h.finalize();
    fr_from_le_bytes_mod_p(out.as_bytes())
}

/// Derive a per-node salt for Merkle hashing:
/// salt = H("MT-SALT" || level || node_idx || seed), mapped to Fr.
pub fn salt_for_node(level: usize, node_idx: usize, seed: &[u8; 32]) -> F {
    let mut h = Hasher::new();
    h.update(b"MT-SALT");
    h.update(&level.to_le_bytes());
    h.update(&node_idx.to_le_bytes());
    h.update(seed);
    let out = h.finalize();
    fr_from_le_bytes_mod_p(out.as_bytes())
}

/// Domain-separation tag for Merkle hashing based on arity.
pub fn ds_tag_for_arity(arity: usize) -> F {
    fr_from_hash("MT-DS", format!("arity-{arity}").as_bytes())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ByteSize(pub usize);

impl core::fmt::Display for ByteSize {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let n = self.0 as f64;
        let (value, suffix) = if n >= (1 << 30) as f64 {
            (n / (1 << 30) as f64, "GiB")
        } else if n >= (1 << 20) as f64 {
            (n / (1 << 20) as f64, "MiB")
        } else if n >= (1 << 10) as f64 {
            (n / (1 << 10) as f64, "KiB")
        } else {
            (n, "B")
        };
        write!(f, "{value:.2} {suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fr_hash_deterministic() {
        let a = fr_from_hash("TAG", b"data");
        let b = fr_from_hash("TAG", b"data");
        assert_eq!(a, b);
        let c = fr_from_hash("TAG", b"data2");
        assert_ne!(a, c);
    }

    #[test]
    fn salt_changes_with_inputs() {
        let seed = [7u8; 32];
        let s1 = salt_for_node(0, 0, &seed);
        let s2 = salt_for_node(0, 1, &seed);
        let s3 = salt_for_node(1, 0, &seed);
        assert_ne!(s1, s2);
        assert_ne!(s1, s3);
        assert_ne!(s2, s3);
    }

    #[test]
    fn ds_tag_depends_on_arity() {
        let t16 = ds_tag_for_arity(16);
        let t32 = ds_tag_for_arity(32);
        assert_ne!(t16, t32);
        // Deterministic
        let t16b = ds_tag_for_arity(16);
        assert_eq!(t16, t16b);
    }

    #[test]
    fn bytesize_display() {
        let b = ByteSize(1536);
        assert_eq!(format!("{b}"), "1.50 KiB");
    }
}
