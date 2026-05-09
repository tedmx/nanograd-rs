pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn relu_derivative(z: f64) -> f64 {
  if z > 0.0 { 1.0 } else { 0.0 }
}
