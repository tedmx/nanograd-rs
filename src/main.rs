mod activations;
mod models;

use models::linear::TinyNet;
use models::simple_mlp::SimpleMLP;
use rand::prelude::SliceRandom;

fn main() {
  println!("=== Linear Net Training ===");
  let mut linear_model = TinyNet::new();
  let data = vec![(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
  
  for _ in 0..500 {
    for (x, y) in &data {
      linear_model.train(*x, *y, 0.01);
    }
  }
  println!("Linear(5.0) -> expected 11.0, got: {:.2}", linear_model.predict(5.0));

  // Simple MLP

  let mut model = SimpleMLP::new(16);
  // Генерируем данные для y = x^2
  let mut train_data: Vec<(f64, f64)> = (-10..=10)
    .map(|i| {
      let x = i as f64 * 0.5;
      (x, x * x)
    })
    .collect();

  println!("Model initialized. Ready for training loop.");

  let epochs = 50000;
  let learning_rate = 0.0001;
  
  let mut rng = rand::thread_rng();

  for epoch in 0..epochs {
    let mut total_loss = 0.0;

    train_data.shuffle(&mut rng);
    for &(x, y_true) in &train_data {
        let pred = model.predict(x);
        total_loss += (pred - y_true).powi(2);
        model.train(x, y_true, learning_rate);
    }

    if epoch % 500 == 0 {
        println!("Epoch {}: Loss {}", epoch, total_loss / train_data.len() as f64);
    }
  }

  println!("\n--- Результаты обучения (Парабола y = x²) ---");
  println!("{:<10} | {:<10} | {:<10}", "X", "Target", "Predict");
  println!("{}", "-".repeat(35));

  // Проверим на более частом шаге, чтобы увидеть плавность
  for i in -5..=5 {
    let x = i as f64; // Берем дробные значения (типа 0.5, 1.5), чтобы проверить обобщение
    let target = x * x;
    let pred = model.predict(x);
    
    let diff = (target - pred).abs();
    let marker = if diff < 0.5 { "✅" } else { "❌" };

    println!("{:<10.1} | {:<10.2} | {:<10.2} {}", x, target, pred, marker);
  }
}
