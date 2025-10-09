use ark_ff::UniformRand;
use ark_pallas::Fr as F;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use rand::{rngs::StdRng, SeedableRng};

// Existing channel-based plain path
use channel::{build_vk_plain, prove_plain, verify_plain};

// DEEP-ALI/FRI APIs live under deep_ali::fri
use deep_ali::fri::{
    deep_fri_prove, deep_fri_verify, deep_fri_proof_size_bytes, AliA, AliE, AliS, AliT, DeepAliMock,
    DeepFriParams, DeepFriProof,
};

fn bench_e2e_plain(c: &mut Criterion) {
    let mut g = c.benchmark_group("e2e_plain");
    for &k in &[12usize, 14, 16] {
        let n = 1usize << k;
        let ds = F::from(2025u64);

        // Setup witness and VK once per input size
        let mut rng = StdRng::seed_from_u64(7);
        let witness: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let vk = build_vk_plain(k, ds);

        // Precompute a proof for verifier bench and report sizes
        let pre_proof = prove_plain(&vk, &witness);

        // If your plain VK/proof types implement serde, keep using bincode.
        // If they do not, these will be 0.
        let vk_size = match bincode::serialize(&vk) {
            Ok(bytes) => bytes.len(),
            Err(_) => 0,
        };
        let proof_size = match bincode::serialize(&pre_proof) {
            Ok(bytes) => bytes.len(),
            Err(_) => 0,
        };
        eprintln!("plain k={} vk={}B proof={}B", k, vk_size, proof_size);

        // Prover time
        g.bench_with_input(BenchmarkId::new("prove", k), &k, |b, &_k| {
            b.iter_batched(
                || (),
                |_| {
                    let proof = prove_plain(&vk, &witness);
                    criterion::black_box(proof);
                },
                BatchSize::SmallInput,
            )
        });

        // Verifier time
        g.bench_with_input(BenchmarkId::new("verify", k), &k, |b, &_k| {
            b.iter(|| {
                let ok = verify_plain(&vk, &pre_proof);
                assert!(ok);
            })
        });
    }
    g.finish();
}

// Toggle presets for MF-FRI
const USE_PAPER_PRESET: bool = false; // set to true to try schedule [16,16,8], r=32

fn bench_e2e_mf_fri(c: &mut Criterion) {
    let mut g = c.benchmark_group("e2e_mf_fri");

    // Lighter default preset to avoid long runs that look like hangs.
    let (schedule, r) = if USE_PAPER_PRESET {
        (vec![16usize, 16usize, 8usize], 32usize)
    } else {
        (vec![8usize, 8usize, 4usize], 8usize)
    };

    // MF-specific tuning to avoid Criterion "unable to complete N samples" warning
    g.measurement_time(std::time::Duration::from_secs(20));
    g.sample_size(10);

    let seed_z = 0xDEEF_BAAD_u64;

    for &k in &[12usize, 14] {
        let n0 = 1usize << k;
        g.throughput(Throughput::Elements(n0 as u64));

        // Build a mock DEEP-ALI witness (replace with a real builder if available)
        let mut rng = StdRng::seed_from_u64(1337);
        let a: AliA = (0..n0).map(|_| F::rand(&mut rng)).collect();
        let s: AliS = (0..n0).map(|_| F::rand(&mut rng)).collect();
        let e: AliE = (0..n0).map(|_| F::rand(&mut rng)).collect();
        let t: AliT = (0..n0).map(|_| F::rand(&mut rng)).collect();

        let params = DeepFriParams {
            schedule: schedule.clone(),
            r,
            seed_z,
        };
        let builder = DeepAliMock;

        eprintln!(
            "mf-fri setup: k={} (n0={}), schedule={:?}, r={}",
            k, n0, schedule, r
        );

        // Precompute a proof for size/verify benches and report size (no serde)
        eprintln!("mf-fri precompute proof…");
        let pre_proof: DeepFriProof = deep_fri_prove(&builder, &a, &s, &e, &t, n0, &params);
        let proof_size_bytes = deep_fri_proof_size_bytes(&pre_proof);
        eprintln!("mf-fri k={} r={} proof≈{}B", k, r, proof_size_bytes);

        // Prover time
        g.bench_with_input(BenchmarkId::new("prove", k), &k, |b, &_k| {
            b.iter_batched(
                || (),
                |_| {
                    let proof = deep_fri_prove(&builder, &a, &s, &e, &t, n0, &params);
                    criterion::black_box(proof);
                },
                BatchSize::SmallInput,
            )
        });

        // Verifier time
        eprintln!("mf-fri precompute verify warmup…");
        assert!(deep_fri_verify(&params, &pre_proof));
        g.bench_with_input(BenchmarkId::new("verify", k), &k, |b, &_k| {
            b.iter(|| {
                let ok = deep_fri_verify(&params, &pre_proof);
                assert!(ok);
            })
        });
    }

    g.finish();
}

criterion_group!(e2e, bench_e2e_plain, bench_e2e_mf_fri);
criterion_main!(e2e);