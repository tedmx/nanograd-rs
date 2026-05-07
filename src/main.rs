use ndarray::Array2;
use rand::Rng;

struct TinyNet {
    w: f64, // Вес
    b: f64, // Смещение (bias)
}

impl TinyNet {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        // Инициализируем случайными числами
        Self {
            w: rng.gen_range(-1.0..1.0),
            b: rng.gen_range(-1.0..1.0),
        }
    }

    // Прямой проход (Forward pass)
    fn predict(&self, x: f64) -> f64 {
        x * self.w + self.b
    }

    fn train(&mut self, x: f64, target: f64, learning_rate: f64) {
        let pred = self.predict(x);
        
        // Ошибка (Loss)
        let error = pred - target;

        // Вычисляем градиенты (производные)
        // Производная ошибки по w: 2 * error * x
        // Производная ошибки по b: 2 * error
        let dw = error * x; 
        let db = error;

        // Обновляем параметры (шагаем против градиента)
        self.w -= learning_rate * dw;
        self.b -= learning_rate * db;
    }
}

fn main() {
    let mut net = TinyNet::new();
    let learning_rate = 0.01;
    
    // Обучающие данные: x и результат (x * 2)
    let training_data = vec![
        (1.0, 2.0),
        (2.0, 4.0),
        (3.0, 6.0),
        (4.0, 8.0),
    ];

    println!("До обучения (x=5): {:.2}", net.predict(5.0));

    // Цикл обучения (эпохи)
    for epoch in 0..1000 {
        for &(x, y) in &training_data {
            net.train(x, y, learning_rate);
        }
        
        if epoch % 200 == 0 {
            let sample_error = (net.predict(5.0) - 10.0).powi(2);
            println!("Эпоха {}: Ошибка на 5.0 = {:.6}", epoch, sample_error);
        }
    }

    println!("---");
    println!("После обучения (x=5): {:.2}", net.predict(5.0));
    println!("Итоговый вес (w): {:.2}, смещение (b): {:.2}", net.w, net.b);
}
