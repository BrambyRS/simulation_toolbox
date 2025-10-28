pub mod erk;

pub trait Model {
    fn name(&self) -> &str;
    fn n_x(&self) -> usize;
    fn n_u(&self) -> usize;

    fn fun(&self, x: &Vec<f64>, u: &Vec<f64>, t: f64) -> Vec<f64>; // Dynamics function
    fn jac(&self, x: &Vec<f64>, u: &Vec<f64>, t: f64) -> Vec<f64>; // Jacobian of f w.r.t x, u, t
}
