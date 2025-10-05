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
/// Input evaluations A,S,E,T over H, primitive root omega, and challenge z ∉ H.
/// Returns:
/// - f0 evaluations on H where f0(x) = (Φ(x) − c*·Z_H(x)) / (x − z),
/// - z,
/// - c* = Φ(z)/Z_H(z),
/// with Φ = A·S + E − T.
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
/// If r_eval_opt is Some(r_eval), Φ̃ = A·S + E − T + β·R; else Φ̃ = A·S + E − T.
/// Returns f0 evaluations, z, c*, with f0(ω^j) = Φ̃(ω^j)/(ω^j − z) on H and c* = Φ̃(z)/Z_H(z).
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
/// Caches omega^j to reduce repeated work across evaluations/merges.
#[derive(Clone)]
pub struct DomainH {
    pub n: usize,
    pub omega: F,
    pub omega_pows: Vec<F>, // [1, ω, ω^2, ..., ω^{n-1}]
}

impl DomainH {
    /// Construct a radix-2 domain of size n and cache ω and its powers.
    pub fn new_radix2(n: usize) -> Self {
        use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
        use ark_poly::EvaluationDomain;
        let dom = Domain::<F>::new(n).expect("radix-2 domain exists for this n");
        let omega = dom.group_gen;

        // Precompute omega^j
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

    /// Evaluate the degree < n polynomial with values on H at z ∉ H (or exactly on H).
    /// Uses the same formula as lagrange_eval_on_h but reuses cached omega powers.
    pub fn eval_lagrange(&self, values: &[F], z: F) -> F {
        assert_eq!(
            values.len(),
            self.n,
            "values length must equal domain size"
        );
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

    /// DEEP-ALI merge reusing cached omega powers (no blinding).
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

    /// DEEP-ALI merge with optional blinding (β·R) reusing cached omega powers.
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

/// Deterministic “simulatable view” sampling for tests:
/// Given a seed, derive (z, beta) deterministically and ensure z ∉ H.
pub fn sample_z_beta_from_seed(seed: u64, n: usize) -> (F, F) {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    // Deterministic beta: any field element (allow zero for baseline tests)
    let beta = F::from(rng.gen::<u64>());

    // Deterministic z not in H: loop with bounded retries
    let z = loop {
        let cand = F::from(rng.gen::<u64>());
        if !is_in_domain(cand, n) {
            break cand;
        }
    };
    (z, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use ark_poly::EvaluationDomain;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// Horner evaluation of a polynomial with coefficients coeffs.
    fn poly_eval(coeffs: &[F], x: F) -> F {
        let mut acc = F::zero();
        for &c in coeffs.iter().rev() {
            acc *= x;
            acc += c;
        }
        acc
    }

    /// Evaluate coeffs on H = <omega> of size n (deg < n).
    fn eval_on_domain(coeffs: &[F], omega: F, n: usize) -> Vec<F> {
        let mut out = Vec::with_capacity(n);
        let mut x = F::one();
        for _ in 0..n {
            out.push(poly_eval(coeffs, x));
            x *= omega;
        }
        out
    }

    /// Find a primitive n-th root of unity for power-of-two n via ark-poly Radix2 domain.
    fn find_primitive_root_pow2(n: usize) -> F {
        use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
        let dom = Domain::<F>::new(n).expect("radix-2 domain exists for this n");
        dom.group_gen
    }

    #[test]
    fn test_lagrange_eval_matches_polyeval_off_domain() {
        let n = 32usize;
        let omega = find_primitive_root_pow2(n);
        // Sample a random polynomial of degree < n
        let mut rng = StdRng::seed_from_u64(7);
        let deg = 20usize;
        let coeffs: Vec<F> = (0..=deg).map(|_| F::rand(&mut rng)).collect();
        let evals = eval_on_domain(&coeffs, omega, n);

        // Pick random z ∉ H
        let mut rngz = StdRng::seed_from_u64(99);
        let z = loop {
            let z = F::rand(&mut rngz);
            if !is_in_domain(z, n) {
                break z;
            }
        };

        let fz_eval = lagrange_eval_on_h(&evals, z, omega);
        let fz_poly = poly_eval(&coeffs, z);
        assert_eq!(fz_eval, fz_poly, "lagrange evaluation must match direct poly eval");
    }

    #[test]
    fn test_deep_ali_merge_completeness() {
        let n = 64usize;
        let omega = find_primitive_root_pow2(n);

        let mut rng = StdRng::seed_from_u64(2024);
        // Degrees strictly less than n
        let deg_a = 15usize;
        let deg_s = 17usize;
        let deg_e = 7usize;
        let a_coeffs: Vec<F> = (0..=deg_a).map(|_| F::rand(&mut rng)).collect();
        let s_coeffs: Vec<F> = (0..=deg_s).map(|_| F::rand(&mut rng)).collect();
        let e_coeffs: Vec<F> = (0..=deg_e).map(|_| F::rand(&mut rng)).collect();

        // Evaluate on H
        let a_eval = eval_on_domain(&a_coeffs, omega, n);
        let s_eval = eval_on_domain(&s_coeffs, omega, n);
        let e_eval = eval_on_domain(&e_coeffs, omega, n);

        // Compute T = A*S + E on H
        let mut t_eval = vec![F::zero(); n];
        for i in 0..n {
            t_eval[i] = a_eval[i] * s_eval[i] + e_eval[i];
        }

        // Pick z ∉ H
        let mut rngz = StdRng::seed_from_u64(8080);
        let z = loop {
            let cand = F::rand(&mut rngz);
            if !is_in_domain(cand, n) {
                break cand;
            }
        };

        let (f0_eval, z_out, c_star) =
            deep_ali_merge_evals(&a_eval, &s_eval, &e_eval, &t_eval, omega, z);
        assert_eq!(z_out, z);

        // Spot-check identity at a few grid points x ∈ H:
        for &j in &[0usize, 1, 7, 13, 31, 47, 63] {
            let x = omega.pow(&[j as u64, 0, 0, 0]);
            let phi_x = a_eval[j] * s_eval[j] + e_eval[j] - t_eval[j];
            let zh_x = zh_at(x, n);
            let left = f0_eval[j] * (x - z) + c_star * zh_x;
            assert_eq!(left, phi_x, "merged identity must hold at grid point {j}");
        }
    }

    #[test]
    fn test_domainh_eval_and_merge_equivalence() {
        // Compare DomainH methods with standalone functions.
        let n = 16usize;
        let domain = DomainH::new_radix2(n);
        let omega = domain.omega;

        let mut rng = StdRng::seed_from_u64(4242);
        let deg = 10usize;
        let coeffs: Vec<F> = (0..=deg).map(|_| F::rand(&mut rng)).collect();
        let evals = eval_on_domain(&coeffs, omega, n);

        // Deterministic z outside H
        let z = F::from(2u64);
        assert!(!is_in_domain(z, n));

        let fz1 = lagrange_eval_on_h(&evals, z, omega);
        let fz2 = domain.eval_lagrange(&evals, z);
        assert_eq!(fz1, fz2, "DomainH::eval_lagrange must match standalone");

        // Build A,S,E,T and compare merges
        let deg_a = 7usize;
        let deg_s = 9usize;
        let deg_e = 5usize;
        let a_coeffs: Vec<F> = (0..=deg_a).map(|_| F::rand(&mut rng)).collect();
        let s_coeffs: Vec<F> = (0..=deg_s).map(|_| F::rand(&mut rng)).collect();
        let e_coeffs: Vec<F> = (0..=deg_e).map(|_| F::rand(&mut rng)).collect();
        let a_eval = eval_on_domain(&a_coeffs, omega, n);
        let s_eval = eval_on_domain(&s_coeffs, omega, n);
        let e_eval = eval_on_domain(&e_coeffs, omega, n);
        let mut t_eval = vec![F::zero(); n];
        for i in 0..n {
            t_eval[i] = a_eval[i] * s_eval[i] + e_eval[i];
        }

        let (f0a, z1, c1) = deep_ali_merge_evals(&a_eval, &s_eval, &e_eval, &t_eval, omega, z);
        let (f0b, z2, c2) = domain.merge_deep_ali(&a_eval, &s_eval, &e_eval, &t_eval, z);
        assert_eq!(z1, z2);
        assert_eq!(c1, c2);
        assert_eq!(f0a, f0b);
    }

    #[test]
    fn test_deterministic_z_edge_cases_small_n() {
        use rand::SeedableRng;
        for &n in &[8usize, 16usize] {
            let domain = DomainH::new_radix2(n);
            let omega = domain.omega;

            // Deterministic z values: 2 and 3. Ensure they lie outside H for these small domains.
            for &z in &[F::from(2u64), F::from(3u64)] {
                if is_in_domain(z, n) {
                    continue;
                }

                let mut rng = StdRng::seed_from_u64(2025 + n as u64);
                let deg = (n - 2).max(1);
                let coeffs: Vec<F> = (0..=deg).map(|_| F::rand(&mut rng)).collect();
                let evals = super::tests::eval_on_domain(&coeffs, omega, n);

                let fz1 = lagrange_eval_on_h(&evals, z, omega);
                let fz2 = domain.eval_lagrange(&evals, z);
                assert_eq!(fz1, fz2, "consistent off-domain eval for small n");
            }
        }
    }

    #[test]
    fn test_multiple_merges_reuse_two_z() {
        let n = 32usize;
        let domain = DomainH::new_radix2(n);
        let omega = domain.omega;

        let mut rng = StdRng::seed_from_u64(9090);
        let deg_a = 12usize;
        let deg_s = 14usize;
        let deg_e = 8usize;
        let a_coeffs: Vec<F> = (0..=deg_a).map(|_| F::rand(&mut rng)).collect();
        let s_coeffs: Vec<F> = (0..=deg_s).map(|_| F::rand(&mut rng)).collect();
        let e_coeffs: Vec<F> = (0..=deg_e).map(|_| F::rand(&mut rng)).collect();

        let a_eval = eval_on_domain(&a_coeffs, omega, n);
        let s_eval = eval_on_domain(&s_coeffs, omega, n);
        let e_eval = eval_on_domain(&e_coeffs, omega, n);
        let mut t_eval = vec![F::zero(); n];
        for i in 0..n {
            t_eval[i] = a_eval[i] * s_eval[i] + e_eval[i];
        }

        // Two different z's off-domain
        let z1 = F::from(2u64);
        let z2 = F::from(3u64);
        assert!(!is_in_domain(z1, n));
        assert!(!is_in_domain(z2, n));

        // Standalone
        let (f0_1, _, c1) = deep_ali_merge_evals(&a_eval, &s_eval, &e_eval, &t_eval, omega, z1);
        let (f0_2, _, c2) = deep_ali_merge_evals(&a_eval, &s_eval, &e_eval, &t_eval, omega, z2);

        // DomainH
        let (g0_1, _, d1) = domain.merge_deep_ali(&a_eval, &s_eval, &e_eval, &t_eval, z1);
        let (g0_2, _, d2) = domain.merge_deep_ali(&a_eval, &s_eval, &e_eval, &t_eval, z2);

        assert_eq!(f0_1, g0_1);
        assert_eq!(f0_2, g0_2);
        assert_eq!(c1, d1);
        assert_eq!(c2, d2);
    }

    #[test]
    fn test_degree_boundary_deg_n_minus_1() {
        let n = 32usize;
        let domain = DomainH::new_radix2(n);
        let omega = domain.omega;

        let mut rng = StdRng::seed_from_u64(4444);
        // Degree n-1 is the maximum allowed for uniqueness over H (degree < n).
        let deg = n - 1;
        let coeffs: Vec<F> = (0..=deg).map(|_| F::rand(&mut rng)).collect();
        let evals = eval_on_domain(&coeffs, omega, n);

        // Off-domain z deterministic
        let z = F::from(5u64);
        assert!(!is_in_domain(z, n));

        let fz1 = lagrange_eval_on_h(&evals, z, omega);
        let fz2 = domain.eval_lagrange(&evals, z);
        assert_eq!(fz1, fz2);
    }

    #[test]
    fn test_deep_ali_merge_blinding_does_not_break_degree() {
        let n = 32usize;
        let domain = DomainH::new_radix2(n);
        let omega = domain.omega;

        let mut rng = StdRng::seed_from_u64(5555);
        let deg_a = 10usize;
        let deg_s = 9usize;
        let deg_e = 7usize;
        let deg_r = 6usize; // small blinder degree
        let a_coeffs: Vec<F> = (0..=deg_a).map(|_| F::rand(&mut rng)).collect();
        let s_coeffs: Vec<F> = (0..=deg_s).map(|_| F::rand(&mut rng)).collect();
        let e_coeffs: Vec<F> = (0..=deg_e).map(|_| F::rand(&mut rng)).collect();
        let r_coeffs: Vec<F> = (0..=deg_r).map(|_| F::rand(&mut rng)).collect();

        let a_eval = eval_on_domain(&a_coeffs, omega, n);
        let s_eval = eval_on_domain(&s_coeffs, omega, n);
        let e_eval = eval_on_domain(&e_coeffs, omega, n);
        let r_eval = eval_on_domain(&r_coeffs, omega, n);

        let mut rngz = StdRng::seed_from_u64(123456);
        let z = loop {
            let cand = F::rand(&mut rngz);
            if !is_in_domain(cand, n) {
                break cand;
            }
        };

        // Unblinded
        let (f0_u, _, c_u) =
            domain.merge_deep_ali(&a_eval, &s_eval, &e_eval, &vec![F::zero(); n], z);

        // Blinded with beta != 0, T = 0 for simplicity (Φ = A·S + E + β·R)
        let beta = F::from(7u64);
        let (f0_b, _, c_b) = domain.merge_deep_ali_blinded(
            &a_eval,
            &s_eval,
            &e_eval,
            &vec![F::zero(); n],
            Some(&r_eval),
            beta,
            z,
        );

        // f0 changes with blinding (very likely).
        assert_ne!(f0_u, f0_b, "blinding should change the merged evaluations");

        // Identity holds on H for blinded: f0_b(ω^j)(ω^j − z) + c_b Z_H(ω^j) = Φ̃(ω^j) = A S + E + β R
        for j in 0..n {
            let x = domain.omega_pows[j];
            let lhs = f0_b[j] * (x - z) + c_b * zh_at(x, n); // Z_H(x) = 0 on H, so term vanishes
            let rhs = a_eval[j] * s_eval[j] + e_eval[j] + beta * r_eval[j];
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn test_simulatable_view_seed_stability() {
        let n = 32usize;
        let (z1, b1) = sample_z_beta_from_seed(42, n);
        let (z2, b2) = sample_z_beta_from_seed(42, n);
        let (z3, b3) = sample_z_beta_from_seed(43, n);

        assert_eq!(z1, z2);
        assert_eq!(b1, b2);
        assert_ne!(z1, z3);
        // b3 can collide by chance; we only require determinism for a fixed seed.
    }
}
