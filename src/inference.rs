use std::{collections::HashMap, error::Error};

use ndarray::{s, stack, Array2, Array3, Axis, Ix2};
use ort::{GraphOptimizationLevel, Session, Tensor};

use crate::constants::{ANNOTATIONS_FPS, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP, MODEL_PATH};
use crate::preprocessing::load_audio::get_audio_input;

fn unwrap_output(
    output: Array3<f32>,
    audio_original_length: usize,
    n_overlapping_frames: usize
) -> Array2<f32> {
    let shape = output.shape();
    if shape.len() != 3 {
        panic!("Expected 3D array");
    }

    let n_batches = shape[0];
    let n_times_short = shape[1];
    let n_freqs = shape[2];

    let n_olap = (0.5 * n_overlapping_frames as f32) as usize;
    let trimmed_output = if n_olap > 0 {
        output.slice(s![.., n_olap..n_times_short - n_olap, ..]).to_owned()
    } else {
        output
    };

    let trimmed_clone = trimmed_output.clone();
    let trimmed_shape = trimmed_clone.shape();
    let unwrapped_output = trimmed_output
        .into_shape((n_batches * trimmed_shape[1], n_freqs))
        .unwrap();

    let n_output_frames_original = ((audio_original_length as f32) * (ANNOTATIONS_FPS as f32 / AUDIO_SAMPLE_RATE as f32)).floor() as usize;

    return unwrapped_output.slice(s![..n_output_frames_original, ..]).to_owned();
}

pub fn run_inference(
    audio_path: &str,
) -> Result<HashMap<String, Array2<f32>>, Box<dyn Error>> {
    let n_overlapping_frames = 30;
    let overlap_len = n_overlapping_frames * FFT_HOP;
    let hop_size = AUDIO_N_SAMPLES - overlap_len;

    let (audio_windows, original_length) = get_audio_input(audio_path, overlap_len, hop_size)?;

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(MODEL_PATH)?;

    let mut output: HashMap<String, Vec<Array2<f32>>> = HashMap::from([
        ("contour".to_string(), vec![]),
        ("onset".to_string(), vec![]),
        ("note".to_string(), vec![]),
    ]);

    for window in audio_windows {
        let window = window.insert_axis(Axis(2)).to_owned();
        let input_shape: Vec<i64> = window.shape().iter().map(|&dim| dim as i64).collect();
        let input_data: Vec<f32> = window.into_raw_vec();
        let input_tensor = Tensor::from_array((input_shape, input_data))?;
        let outputs = model.run(ort::inputs![input_tensor]?)?;

        for (&k, v) in outputs.iter() {
            // let test = v.try_extract_tensor::<f32>()?.into_dimensionality::<Ix2>()?.to_owned();
            let test = v
                .try_extract_tensor::<f32>()?
                .index_axis(Axis(0), 0)
                .into_dimensionality::<Ix2>()?
                .to_owned();
            
            if k == "StatefulPartitionedCall:0" {
                output.get_mut("contour").unwrap().push(test)
            } else if k == "StatefulPartitionedCall:1" { 
                output.get_mut("note").unwrap().push(test)
            } else if k == "StatefulPartitionedCall:2" {
                output.get_mut("onset").unwrap().push(test)
            }
        }
    }

    let unwrapped_output: HashMap<String, Array2<f32>> = output.into_iter().map(|(k, v)| {
        let concatenated = stack(Axis(0), &v.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
        let unwrapped = unwrap_output(concatenated, original_length, n_overlapping_frames);
        (k, unwrapped)
    }).collect();

    Ok(unwrapped_output)
}