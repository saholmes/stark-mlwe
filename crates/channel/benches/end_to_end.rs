use ark_pallas::Fr as F;
use ark_ff::UniformRand;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Serialize, Deserialize};
use bincode;

use channel::{
    build_vk_plain, build_vk_mf, prove_plain, verify_plain, prove_mf, verify_mf, VKVariant,
};

fn bench_e2e_plain(c: &mut Criterion) {
    let mut g = c.benchmark_group("e2e_plain");
    for &k in &[12usize, 14, 16] {
        let n = 1usize << k;
        let ds = F::from(2025u64);

        // Setup witness and VK once per input
        let mut rng = StdRng::seed_from_u64(7);
        let witness: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let vk = build_vk_plain(k, ds);

        // Precompute a proof for verifier bench
        let pre_proof = prove_plain(&vk, &witness);
        let vk_size = bincode::serialize(&vk).unwrap().len();
        let proof_size = bincode::serialize(&pre_proof).unwrap().len();
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

fn bench_e2e_mf(c: &mut Criterion) {
    let mut g = c.benchmark_group("e2e_mf");
    let qpr = 1usize;

    for &k in &[12usize, 14] {
        let n = 1usize << k;
        let ds = F::from(6060u64);

        let mut rng = StdRng::seed_from_u64(1337);
        let witness: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let vk = build_vk_mf(k, ds, qpr);

        let pre_proof = prove_mf(&vk, &witness);
        let vk_size = bincode::serialize(&vk).unwrap().len();
        let proof_size = bincode::serialize(&pre_proof).unwrap().len();
        eprintln!("mf k={} q={} vk={}B proof={}B", k, qpr, vk_size, proof_size);

        g.bench_with_input(BenchmarkId::new("prove", k), &k, |b, &_k| {
            b.iter_batched(
                || (),
                |_| {
                    let proof = prove_mf(&vk, &witness);
                    criterion::black_box(proof);
                },
                BatchSize::SmallInput,
            )
        });

        g.bench_with_input(BenchmarkId::new("verify", k), &k, |b, &_k| {
            b.iter(|| {
                let ok = verify_mf(&vk, &pre_proof);
                assert!(ok);
            })
        });
    }
    g.finish();
}

criterion_group!(e2e, bench_e2e_plain, bench_e2e_mf);
criterion_main!(e2e);
