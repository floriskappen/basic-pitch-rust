use ndarray::{s, Array1, Axis};

use crate::constants::{AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE};

pub struct WindowedAudio<'a> {
    audio: &'a Array1<f32>,
    hop_size: usize,
    index: usize,
}

impl<'a> Iterator for WindowedAudio<'a> {
    type Item = (Array1<f32>, (f64, f64));

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.audio.len() {
            return None;
        }

        let end_index = (self.index + AUDIO_N_SAMPLES).min(self.audio.len());
        let mut window = self.audio.slice(s![self.index..end_index]).to_owned();

        if window.len() < AUDIO_N_SAMPLES {
            let padding = Array1::zeros(AUDIO_N_SAMPLES - window.len());
            window = ndarray::concatenate(Axis(0), &[window.view(), padding.view()]).unwrap();
        }

        let t_start = self.index as f64 / AUDIO_SAMPLE_RATE as f64;
        let window_time = (t_start, t_start + (AUDIO_N_SAMPLES as f64 / AUDIO_SAMPLE_RATE as f64));

        self.index += self.hop_size;
        Some((window, window_time))
    }
}

pub fn window_audio_file(audio: &Array1<f32>, hop_size: usize) -> WindowedAudio {
    WindowedAudio {
        audio,
        hop_size,
        index: 0,
    }
}
