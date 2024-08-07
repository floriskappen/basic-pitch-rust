use std::{collections::HashMap, error::Error};

use ndarray::{concatenate, s, Array2, Array3, ArrayView3, Axis, Ix2};
use ort::{GraphOptimizationLevel, Session, Tensor};

use crate::constants::{ANNOTATIONS_FPS, AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP, MODEL_PATH};
use crate::preprocessing::load_audio::get_audio_input;

fn unwrap_output(
    output: Array3<f32>,
    audio_original_length: usize,
    n_overlapping_frames: usize
) -> Option<Array2<f32>> {
    let shape = output.shape();
    if shape.len() != 3 {
        return None;
    }

    let n_olap = (0.5 * n_overlapping_frames as f32) as usize;
    let trimmed_output = if n_olap > 0 {
        output.slice(s![.., n_olap..shape[1] - n_olap, ..]).to_owned()
    } else {
        output.clone()
    };

    let unwrapped_output = trimmed_output
        .into_shape((shape[0] * (shape[1] - 2 * n_olap), shape[2]))
        .unwrap();

    let n_output_frames_original = ((audio_original_length as f32) * (ANNOTATIONS_FPS as f32 / AUDIO_SAMPLE_RATE as f32)).floor() as usize;

    Some(unwrapped_output.slice(s![..n_output_frames_original, ..]).to_owned())
}

pub fn run_inference(
    audio_path: &str,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), Box<dyn Error>> {
    let n_overlapping_frames = 30;
    let overlap_len = n_overlapping_frames * FFT_HOP;
    let hop_size = AUDIO_N_SAMPLES - overlap_len;

    let (audio_windows, original_length) = get_audio_input(audio_path, overlap_len, hop_size)?;

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(MODEL_PATH)?;

    let mut output: HashMap<String, Vec<Array3<f32>>> = HashMap::from([
        ("contours".to_string(), vec![]),
        ("onsets".to_string(), vec![]),
        ("frames".to_string(), vec![]),
    ]);

    for window in audio_windows {
        // Ensure the window has the correct shape
        let window = window.insert_axis(Axis(2)).to_owned();
        let input_shape: Vec<i64> = window.shape().iter().map(|&dim| dim as i64).collect();
        let input_data: Vec<f32> = window.into_raw_vec();
        let input_tensor = Tensor::from_array((input_shape, input_data))?;
        let outputs = model.run(ort::inputs![input_tensor]?)?;
    
        for (&k, v) in outputs.iter() {
            let value = v
                .try_extract_tensor::<f32>()?
                .index_axis(Axis(0), 0)
                .into_dimensionality::<Ix2>()?
                .insert_axis(Axis(0))
                .to_owned();

            if k == "StatefulPartitionedCall:0" {
                output.get_mut("contours").unwrap().push(value.clone())
            } else if k == "StatefulPartitionedCall:1" { 
                output.get_mut("frames").unwrap().push(value.clone())
            } else if k == "StatefulPartitionedCall:2" {
                output.get_mut("onsets").unwrap().push(value.clone())
            }
        }

    }
    
    let unwrapped_output: HashMap<String, Array2<f32>> = output.into_iter().map(|(k, v)| {
        let views: Vec<ArrayView3<f32>> = v.iter().map(|array| array.view()).collect();
        let concatenated = concatenate(Axis(0), views.as_slice()).unwrap();
        let unwrapped = unwrap_output(concatenated, original_length, n_overlapping_frames).unwrap();
        (k, unwrapped)
    }).collect();

    Ok((
        unwrapped_output.get("contours").unwrap().clone(),
        unwrapped_output.get("frames").unwrap().clone(),
        unwrapped_output.get("onsets").unwrap().clone(),
    ))
}