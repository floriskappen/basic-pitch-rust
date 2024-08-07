use std::{error::Error, path::Path};

use hound::WavReader;
use ndarray::{concatenate, Array1, Array2, Axis};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

use crate::constants::AUDIO_SAMPLE_RATE;

use crate::preprocessing::windowed_audio::window_audio_file;

fn load_and_convert_audio<P: AsRef<Path>>(path: P, target_sample_rate: u32) -> Result<(Array1<f32>, usize), Box<dyn Error>> {
    // Read the input WAV file
    let reader = WavReader::open(path)?;
    let mut spec = reader.spec();
    let duration = reader.duration() as usize;

    let max_sample_value = (2.0_f64.powi(spec.bits_per_sample as i32 - 1) - 1.0) as i32;

    let mut reader_samples = reader.into_samples::<i32>();
    let samples;

    // If it's stereo, convert it to mono
    if spec.channels == 2 {
        let mut mono_samples: Vec<i32> = vec![];
        while let (Some(left), Some(right)) = (reader_samples.next(), reader_samples.next()) {
            let left = left?;
            let right = right?;
            let mono_sample = (left as i32 + right as i32) / 2;
            mono_samples.push(mono_sample)
        }

        samples = mono_samples;
        spec.channels = 1
    } else {
        samples = reader_samples.into_iter().map(|s| s.unwrap()).collect();
    }

    let mut channel_data: Vec<Vec<f64>> = vec![Vec::new()];
    for &sample in samples.iter() {
        channel_data[0].push(sample as f64 / max_sample_value as f64);
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let resample_ratio = target_sample_rate as f64 / spec.sample_rate as f64;
    let mut resampler = SincFixedIn::<f64>::new(
        target_sample_rate as f64 / spec.sample_rate as f64,
        2.0,
        params,
        duration,
        1
    )?;
    let channel_resampled_data = resampler.process(&channel_data, None)?;

    // Convert the channel data vector into a single Vec<f32> for further processing
    let resampled_samples_f32: Vec<f32> = channel_resampled_data[0].iter().map(|&s| s as f32).collect();

    Ok((Array1::from(resampled_samples_f32), (duration as f64 * resample_ratio) as usize))
}

pub fn get_audio_input(
    audio_path: &str,
    overlap_len: usize,
    hop_size: usize,
) -> Result<(Vec<Array2<f32>>, usize), Box<dyn Error>> {
    let (audio_original, original_length) = load_and_convert_audio(audio_path, AUDIO_SAMPLE_RATE as u32)?;
    
    // Padding with half the overlap length
    let padding = Array1::zeros(overlap_len / 2);
    let padded_audio = concatenate(Axis(0), &[padding.view(), audio_original.view()])?;

    let mut audio_windows = vec![];
    for (window, _) in window_audio_file(&padded_audio, hop_size) {
        // Expanding dimensions to match tf.expandDims(..., -1)
        let expanded_window = window.insert_axis(Axis(0)).to_owned();
        audio_windows.push(expanded_window);
    }

    Ok((audio_windows, original_length))
}