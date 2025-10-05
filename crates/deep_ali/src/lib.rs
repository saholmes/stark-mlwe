use ark_ff::{Field, One, Zero};
use ark_pallas::Fr as F;

/// Return true if z ∈ H = <omega> of size n, i.e., z^n == 1.
fn is_in_domain(z: F, n: usize) -> bool {
    z.pow(&[n as u64, 0, 0, 0]) == F::one()
}

/// Vanishing polynomial on H: Z_H(z) = z^n - 1
fn zh_at(z: F, n: usize) -> F {
    z.pow(&[n as u64, 0, 0, 0]) - F::one()
}

/// Evaluate the unique degree < n polynomial with values = {f(ω^j)} on H at z ∉ H:
/// Using the correct barycentric form for multiplicative subgroup H:
/// f(z) = (Z_H(z)/n) * sum_j f(ω^j) * ω^j / (z - ω^j)
pub fn lagrange_eval_on_h(values: &[F], z: F, omega: F) -> F {
    let n = values.len();
    assert!(n > 0, "non-empty domain");
    if is_in_domain(z, n) {
        // Exact on-grid lookup
        let mut omega_j = F::one();
        for j in 0..n {
            if z == omega_j {
                return values[j];
            }
            omega_j *= omega;
        }
        panic!("z in domain but not matching a power of omega");
    }

    let zh = zh_at(z, n);
    let n_inv = F::from(n as u64)
        .inverse()
        .expect("n invertible in prime field");

    let mut sum = F::zero();
    let mut omega_j = F::one(); // ω^0
    for j in 0..n {
        let inv = (z - omega_j).inverse().expect("z ∉ H");
        sum += values[j] * omega_j * inv;
        omega_j *= omega;
    }
    zh * n_inv * sum
}

/// DEEP-ALI merge (no blinding).
pub fn deep_ali_merge_evals(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    omega: F,
    z: F,
) -> (Vec<F>, F, F) {
    deep_ali_merge_evals_blinded(a_eval, s_eval, e_eval, t_eval, None, F::zero(), omega, z)
}

/// DEEP-ALI merge with optional blinding term β·R(x).
pub fn deep_ali_merge_evals_blinded(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    r_eval_opt: Option<&[F]>,
    beta: F,
    omega: F,
    z: F,
) -> (Vec<F>, F, F) {
    let n = a_eval.len();
    assert!(n > 1);
    assert_eq!(s_eval.len(), n);
    assert_eq!(e_eval.len(), n);
    assert_eq!(t_eval.len(), n);
    if let Some(r_eval) = r_eval_opt {
        assert_eq!(r_eval.len(), n);
    }
    assert!(!is_in_domain(z, n), "z must be outside H");

    // Φ̃ on H
    let mut phi_eval = vec![F::zero(); n];
    for i in 0..n {
        let base = a_eval[i] * s_eval[i] + e_eval[i] - t_eval[i];
        phi_eval[i] = if let Some(r_eval) = r_eval_opt {
            base + beta * r_eval[i]
        } else {
            base
        };
    }

    // Φ̃(z) via Lagrange
    let phi_z = lagrange_eval_on_h(&phi_eval, z, omega);
    let zh_z = zh_at(z, n);
    let c_star = phi_z * zh_z.inverse().expect("z ∉ H ⇒ Z_H(z) ≠ 0");

    // f0 on H: Z_H(ω^j)=0 ⇒ f0(ω^j) = Φ̃(ω^j)/(ω^j − z)
    let mut f0_eval = vec![F::zero(); n];
    let mut omega_j = F::one();
    for j in 0..n {
        f0_eval[j] = phi_eval[j] * (omega_j - z).inverse().expect("z ∉ H");
        omega_j *= omega;
    }

    (f0_eval, z, c_star)
}

/// Lightweight domain cache for H = <omega> (radix-2).
#[derive(Clone)]
pub struct DomainH {
    pub n: usize,
    pub omega: F,
    pub omega_pows: Vec<F>, // [1, ω, ω^2, ..., ω^{n-1}]
}

impl DomainH {
    pub fn new_radix2(n: usize) -> Self {
        use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
        use ark_poly::EvaluationDomain;
        let dom = Domain::<F>::new(n).expect("radix-2 domain exists for this n");
        let omega = dom.group_gen;

        let mut omega_pows = Vec::with_capacity(n);
        let mut x = F::one();
        for _ in 0..n {
            omega_pows.push(x);
            x *= omega;
        }

        Self {
            n,
            omega,
            omega_pows,
        }
    }

    pub fn eval_lagrange(&self, values: &[F], z: F) -> F {
        assert_eq!(values.len(), self.n, "values length must equal domain size");
        if is_in_domain(z, self.n) {
            for (j, &w) in self.omega_pows.iter().enumerate() {
                if z == w {
                    return values[j];
                }
            }
            panic!("z in domain but not matching cached omega powers");
        }

        let zh = zh_at(z, self.n);
        let n_inv = F::from(self.n as u64)
            .inverse()
            .expect("n invertible in prime field");

        let mut sum = F::zero();
        for j in 0..self.n {
            let wj = self.omega_pows[j];
            let inv = (z - wj).inverse().expect("z ∉ H");
            sum += values[j] * wj * inv;
        }
        zh * n_inv * sum
    }

    pub fn merge_deep_ali(
        &self,
        a_eval: &[F],
        s_eval: &[F],
        e_eval: &[F],
        t_eval: &[F],
        z: F,
    ) -> (Vec<F>, F, F) {
        self.merge_deep_ali_blinded(a_eval, s_eval, e_eval, t_eval, None, F::zero(), z)
    }

    pub fn merge_deep_ali_blinded(
        &self,
        a_eval: &[F],
        s_eval: &[F],
        e_eval: &[F],
        t_eval: &[F],
        r_eval_opt: Option<&[F]>,
        beta: F,
        z: F,
    ) -> (Vec<F>, F, F) {
        assert_eq!(a_eval.len(), self.n);
        assert_eq!(s_eval.len(), self.n);
        assert_eq!(e_eval.len(), self.n);
        assert_eq!(t_eval.len(), self.n);
        if let Some(r_eval) = r_eval_opt {
            assert_eq!(r_eval.len(), self.n);
        }
        assert!(!is_in_domain(z, self.n), "z must be outside H");

        // Φ̃ on H
        let mut phi_eval = vec![F::zero(); self.n];
        for i in 0..self.n {
            let base = a_eval[i] * s_eval[i] + e_eval[i] - t_eval[i];
            phi_eval[i] = if let Some(r_eval) = r_eval_opt {
                base + beta * r_eval[i]
            } else {
                base
            };
        }

        let phi_z = self.eval_lagrange(&phi_eval, z);
        let zh_z = zh_at(z, self.n);
        let c_star = phi_z * zh_z.inverse().expect("z ∉ H ⇒ Z_H(z) ≠ 0");

        // f0 on H
        let mut f0_eval = vec![F::zero(); self.n];
        for j in 0..self.n {
            f0_eval[j] = phi_eval[j] * (self.omega_pows[j] - z)
                .inverse()
                .expect("z ∉ H");
        }

        (f0_eval, z, c_star)
    }
}

/// Deterministic “simulatable view” sampling for tests.
pub fn sample_z_beta_from_seed(seed: u64, n: usize) -> (F, F) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    let beta = F::from(rng.gen::<u64>());
    let z = loop {
        let cand = F::from(rng.gen::<u64>());
        if !is_in_domain(cand, n) {
            break cand;
        }
    };
    (z, beta)
}

pub mod fri;
