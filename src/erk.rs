use crate::Model;

pub struct ExplicitRK {
    n_stages: usize,
    a: Vec<f64>, // Row-major storage of the A matrix
    b: Vec<f64>, // Coefficients for final combination
    c: Vec<f64>, // Nodes
}

impl ExplicitRK {
    pub fn new(n_stages: usize, a: &Vec<f64>, b: &Vec<f64>, c: &Vec<f64>) -> Self {
        // Validate the lengths of a, b, and c
        assert_eq!(a.len(), n_stages * n_stages);
        assert_eq!(b.len(), n_stages);
        assert_eq!(c.len(), n_stages);
        // Assert A is strictly lower triangular
        for i in 0..n_stages {
            for j in i..n_stages {
                assert_eq!(
                    a[i * n_stages + j],
                    0.0,
                    "A[{},{}] = {}, A is must be strictly lower triangular for explicit RK method",
                    i,
                    j,
                    a[i * n_stages + j]
                );
            }
        }
        return ExplicitRK {
            n_stages,
            a: a.clone(),
            b: b.clone(),
            c: c.clone(),
        };
    }

    // Default solvers
    pub fn euler() -> Self {
        let n_stages: usize = 1;
        let a: Vec<f64> = vec![0.0]; // 1x1 matrix
        let b: Vec<f64> = vec![1.0];
        let c: Vec<f64> = vec![0.0];
        return ExplicitRK::new(n_stages, &a, &b, &c);
    }

    pub fn rk4() -> Self {
        let n_stages: usize = 4;
        let a: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, // Row 1
            0.5, 0.0, 0.0, 0.0, // Row 2
            0.0, 0.5, 0.0, 0.0, // Row 3
            0.0, 0.0, 1.0, 0.0, // Row 4
        ];
        let b: Vec<f64> = vec![1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
        let c: Vec<f64> = vec![0.0, 0.5, 0.5, 1.0];
        return ExplicitRK::new(n_stages, &a, &b, &c);
    }

    // Getter methods
    pub fn n_stages(&self) -> usize {
        return self.n_stages;
    }

    pub fn a(&self) -> &Vec<f64> {
        return &self.a;
    }

    pub fn b(&self) -> &Vec<f64> {
        return &self.b;
    }

    pub fn c(&self) -> &Vec<f64> {
        return &self.c;
    }

    // Forward simulation step
    pub fn step(
        &self,
        model: &impl Model,
        x0: &Vec<f64>,
        u: &Vec<f64>,
        t: f64,
        h: f64,
    ) -> Vec<f64> {
        let n_x: usize = model.n_x();
        let n_u: usize = model.n_u();
        assert_eq!(
            x0.len(),
            n_x,
            "x0 has length {}, expected {}",
            x0.len(),
            n_x
        );
        assert_eq!(u.len(), n_u, "u has length {}, expected {}", u.len(), n_u);

        let mut k: Vec<f64> = vec![0.0; n_x * self.n_stages];

        for i in 0..self.n_stages {
            let mut x_temp: Vec<f64> = x0.clone();
            for j in 0..i {
                let a_ij: f64 = self.a[i * self.n_stages + j];
                for xi in 0..n_x {
                    x_temp[xi] += h * a_ij * k[j * n_x + xi];
                }
            }
            let t_i: f64 = t + self.c[i] * h;
            let k_i: Vec<f64> = model.fun(&x_temp, u, t_i);
            for xi in 0..n_x {
                k[i * n_x + xi] = k_i[xi];
            }
        }

        let mut x_next: Vec<f64> = x0.clone();
        for i in 0..self.n_stages {
            let b_i: f64 = self.b[i];
            for xi in 0..n_x {
                x_next[xi] += h * b_i * k[i * n_x + xi];
            }
        }

        return x_next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;

    // Test creation of ExplicitRK methods
    #[test]
    fn test_erk_euler_manual() {
        // Euler method (1-stage explicit RK)
        let n_stages: usize = 1;
        let a: Vec<f64> = vec![0.0]; // 1x1 matrix
        let b: Vec<f64> = vec![1.0];
        let c: Vec<f64> = vec![0.0];

        let erk = ExplicitRK::new(n_stages, &a, &b, &c);
        assert_eq!(erk.n_stages, 1);
        assert_eq!(erk.a, vec![0.0]);
        assert_eq!(erk.b, vec![1.0]);
        assert_eq!(erk.c, vec![0.0]);
    }

    #[test]
    fn test_erk_euler() {
        // Euler method (1-stage explicit RK)
        let erk = ExplicitRK::euler();
        assert_eq!(erk.n_stages(), 1);
        assert_eq!(erk.a(), &vec![0.0]);
        assert_eq!(erk.b(), &vec![1.0]);
        assert_eq!(erk.c(), &vec![0.0]);
    }

    #[test]
    fn test_erk_rk4_manual() {
        // Classical RK4 method (4-stage explicit RK)
        let n_stages: usize = 4;
        let a: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, // Row 1
            0.5, 0.0, 0.0, 0.0, // Row 2
            0.0, 0.5, 0.0, 0.0, // Row 3
            0.0, 0.0, 1.0, 0.0, // Row 4
        ];
        let b: Vec<f64> = vec![1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
        let c: Vec<f64> = vec![0.0, 0.5, 0.5, 1.0];

        let erk = ExplicitRK::new(n_stages, &a, &b, &c);
        assert_eq!(erk.n_stages(), n_stages);
        assert_eq!(erk.a(), &a);
        assert_eq!(erk.b(), &b);
        assert_eq!(erk.c(), &c);
    }

    #[test]
    fn test_erk_rk4() {
        // Classical RK4 method (4-stage explicit RK)
        let n_stages: usize = 4;
        let a: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, // Row 1
            0.5, 0.0, 0.0, 0.0, // Row 2
            0.0, 0.5, 0.0, 0.0, // Row 3
            0.0, 0.0, 1.0, 0.0, // Row 4
        ];
        let b: Vec<f64> = vec![1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
        let c: Vec<f64> = vec![0.0, 0.5, 0.5, 1.0];

        let erk = ExplicitRK::rk4();
        assert_eq!(erk.n_stages(), n_stages);
        assert_eq!(erk.a(), &a);
        assert_eq!(erk.b(), &b);
        assert_eq!(erk.c(), &c);
    }

    // Test invalid ExplicitRK creations
    #[test]
    fn test_irk_euler() {
        // Test that creating an IRK with non-lower-triangular A fails
        let n_stages: usize = 1;
        let a: Vec<f64> = vec![1.0];
        let b: Vec<f64> = vec![1.0];
        let c: Vec<f64> = vec![1.0];

        let result = std::panic::catch_unwind(|| {
            ExplicitRK::new(n_stages, &a, &b, &c);
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_sizes() {
        // Test that creating an ERK with invalid A dimensions fails
        let n_stages: usize = 2;
        let a: Vec<f64> = vec![0.0, 0.0]; // Invalid size
        let b: Vec<f64> = vec![1.0, 0.0];
        let c: Vec<f64> = vec![0.0, 0.5];

        let result = std::panic::catch_unwind(|| {
            ExplicitRK::new(n_stages, &a, &b, &c);
        });
        assert!(result.is_err());

        // Test that creating an ERK with invalid b size fails
        let a: Vec<f64> = vec![0.0, 0.0, 0.5, 0.0];
        let b: Vec<f64> = vec![1.0]; // Invalid size
        let result = std::panic::catch_unwind(|| {
            ExplicitRK::new(n_stages, &a, &b, &c);
        });
        assert!(result.is_err());

        // Test that creating an ERK with invalid c size fails
        let b: Vec<f64> = vec![1.0, 0.0];
        let c: Vec<f64> = vec![0.0]; // Invalid size
        let result = std::panic::catch_unwind(|| {
            ExplicitRK::new(n_stages, &a, &b, &c);
        });
        assert!(result.is_err());
    }

    struct SpringMassDamper {
        m: f64,
        k: f64,
        c: f64,
    }

    impl Model for SpringMassDamper {
        fn name(&self) -> &str {
            return "Spring Mass Damper";
        }

        fn n_x(&self) -> usize {
            return 2;
        }

        fn n_u(&self) -> usize {
            return 1;
        }

        fn fun(&self, x: &Vec<f64>, u: &Vec<f64>, _t: f64) -> Vec<f64> {
            let mut dxdt: Vec<f64> = vec![0.0; 2];
            dxdt[0] = x[1];
            dxdt[1] = (u[0] - self.c * x[1] - self.k * x[0]) / self.m;
            return dxdt;
        }

        fn jac(&self, _x: &Vec<f64>, _u: &Vec<f64>, _t: f64) -> Vec<f64> {
            // Not implemented for this test
            vec![0.0; 4]
        }
    }

    #[test]
    fn test_erk_euler_step() {
        let model = SpringMassDamper {
            m: 1.0,
            k: 1.0,
            c: 1.0,
        };
        let erk: ExplicitRK = ExplicitRK::euler();

        let x0: Vec<f64> = vec![1.0, 0.0]; // Initial state: position=1, velocity=0
        let u: Vec<f64> = vec![0.0]; // No external force
        let t: f64 = 0.0;
        let h: f64 = 0.1; // Time step

        let x1: Vec<f64> = erk.step(&model, &x0, &u, t, h);
        let expected_x1: Vec<f64> = vec![1.0, -0.1]; // Hand calculated

        for i in 0..x1.len() {
            assert!(
                (x1[i] - expected_x1[i]).abs() < 1e-6,
                "x1[{}] = {}, expected {}",
                i,
                x1[i],
                expected_x1[i]
            );
        }

        let x2 = erk.step(&model, &x1, &u, t + h, h);
        let expected_x2: Vec<f64> = vec![0.99, -0.1 + (-1.0 + 0.1) * h]; // Hand calculated

        for i in 0..x2.len() {
            assert!(
                (x2[i] - expected_x2[i]).abs() < 1e-6,
                "x2[{}] = {}, expected {}",
                i,
                x2[i],
                expected_x2[i]
            );
        }
    }
}
