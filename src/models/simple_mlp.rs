use crate::activations::{relu_derivative};
use rand::Rng;

pub struct SimpleMLP {
  // Слой 1 (скрытый): 4 нейрона
  w1: Vec<f64>,
  b1: Vec<f64>,
  // Слой 2 (выходной): 1 нейрон
  w2: Vec<f64>,
  b2: f64,
}

impl SimpleMLP {
  pub fn new(hidden_size: usize) -> Self {
    let mut rng = rand::thread_rng();
    // Инициализация вектора случайными числами
    let w1 = (0..hidden_size).map(|_| rng.gen_range(-0.2..0.2)).collect();
    let b1 = (0..hidden_size).map(|_| 0.0).collect();
    let w2 = (0..hidden_size).map(|_| rng.gen_range(-0.2..0.2)).collect();
    
    Self {
      w1,
      b1,
      w2,
      b2: 0.0,
    }
  }

  pub fn predict(&self, x: f64) -> f64 {
    let mut sum = 0.0;
    
    for i in 0..self.w1.len() {
      // Линейная часть для скрытого нейрона
      let z = x * self.w1[i] + self.b1[i];
      // Нелинейный "излом" (ReLU)
      let activated = if z > 0.0 { z } else { 0.0 };
      // Накопление вклада в финальный результат
      sum += activated * self.w2[i];
    }
    
    sum + self.b2
  }

  // Внутри impl SimpleMLP в src/models/simple_mlp.rs

  pub fn train(&mut self, x: f64, target: f64, learning_rate: f64) {
    // --- 1. Forward Pass (с сохранением промежуточных данных) ---
    let mut hidden_inputs = Vec::with_capacity(self.w1.len());
    let mut hidden_outputs = Vec::with_capacity(self.w1.len());

    for i in 0..self.w1.len() {
      let z = x * self.w1[i] + self.b1[i];
      hidden_inputs.push(z);
      hidden_outputs.push(if z > 0.0 { z } else { 0.0 });
    }

    let prediction = hidden_outputs.iter()
      .zip(&self.w2)
      .map(|(h, w)| h * w)
      .sum::<f64>() + self.b2;

    // --- 2. Вычисление ошибки на выходе ---
    // Ошибка (Loss) = Prediction - Target
    let output_error = prediction - target;

    // --- 3. Обратное распространение на выходной слой ---
    // Обновляем b2 и веса w2
    self.b2 -= learning_rate * output_error;
    
    for i in 0..self.w2.len() {
      // Вес w2[i] виноват пропорционально значению, которое пришло из скрытого слоя
      let dw2 = output_error * hidden_outputs[i];
      self.w2[i] -= learning_rate * dw2;
    }

    // --- 4. Обратное распространение на скрытый слой ---
    for i in 0..self.w1.len() {
      // "Доля вины" скрытого нейрона: ошибка выхода * вес связи
      let hidden_error = output_error * self.w2[i];
      
      // Умножаем на производную ReLU, чтобы учесть "шарнир"
      let gradient = hidden_error * relu_derivative(hidden_inputs[i]);
      
      // Обновляем веса первого слоя (w1 зависит от входа x)
      self.w1[i] -= learning_rate * gradient * x;
      self.b1[i] -= learning_rate * gradient;
    }
  }
}
