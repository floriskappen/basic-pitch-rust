use std::{error::Error, path::Path};

use hound::WavReader;
use ndarray::{Array1, Array2, Axis};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType};

use crate::constants::AUDIO_SAMPLE_RATE;

use crate::preprocessing::windowed_audio::window_audio_file;

fn load_audio<P: AsRef<Path>>(path: P, target_sample_rate: u32) -> Result<Array1<f32>, Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
    // Convert to mono if needed
    let mono_samples = if spec.channels == 2 {
        let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        let mut mono_samples = vec![];
        for chunk in samples.chunks(2) {
            let left = chunk[0];
            let right = chunk[1];
            let mono_sample = ((left as i32 + right as i32) / 2) as f32;
            mono_samples.push(mono_sample);
        }
        mono_samples
    } else {
        samples.into_iter().map(|s| s as f32).collect()
    };

    // Resample if needed
    let current_sample_rate = spec.sample_rate;

    let resampled_samples = if current_sample_rate != target_sample_rate {
        let resample_ratio = target_sample_rate as f64 / current_sample_rate as f64;

        // Sinc interpolation parameters that match librosa's quality settings
        let sinc_len = 64;  // This can be adjusted based on desired quality
        let f_cutoff = 0.95;  // Cutoff frequency to avoid aliasing
        let interpolation = SincInterpolationType::Cubic;
        let oversampling_factor = 256;  // Higher value for better quality
        let window = rubato::WindowFunction::BlackmanHarris2;  // Blackman-Harris window function
    
        let mut resampler = SincFixedIn::<f32>::new(
            resample_ratio,
            2.0,
            SincInterpolationParameters {
                sinc_len,
                f_cutoff,
                interpolation,
                oversampling_factor,
                window,
            },
            mono_samples.len(),
            1,
        )?;

        // Process the entire mono_samples at once
        let mut output = resampler.process(&[&mono_samples], None)?;
        output.pop().unwrap()  // We only have one channel, so we get the first item
    } else {
        mono_samples
    };

    Ok(Array1::from(resampled_samples))
}

pub fn get_audio_input(
    audio_path: &str,
    overlap_len: usize,
    hop_size: usize,
) -> Result<(Vec<Array2<f32>>, usize), Box<dyn Error>> {
    let audio_original = load_audio(audio_path, AUDIO_SAMPLE_RATE as u32)?;
    let original_length = audio_original.len();
    let padding = Array1::zeros(overlap_len / 2);
    let padded_audio = ndarray::concatenate(Axis(0), &[padding.view(), audio_original.view()])?;

    let mut audio_windows = vec![];
    for (window, _) in window_audio_file(&padded_audio, hop_size) {
        audio_windows.push(window.insert_axis(Axis(0)));
    }

    Ok((audio_windows, original_length))
}
