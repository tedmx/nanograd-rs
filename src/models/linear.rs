use rand::Rng;

pub struct TinyNet {
    w: f64, // Вес
    b: f64, // Смещение (bias)
}

impl TinyNet {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        // Инициализируем случайными числами
        Self {
            w: rng.gen_range(-1.0..1.0),
            b: rng.gen_range(-1.0..1.0),
        }
    }

    // Прямой проход (Forward pass)
    pub fn predict(&self, x: f64) -> f64 {
        x * self.w + self.b
    }

    pub fn train(&mut self, x: f64, target: f64, learning_rate: f64) {
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
