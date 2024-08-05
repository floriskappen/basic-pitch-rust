use crate::{constants::{ANNOTATIONS_BASE_FREQUENCY, CONTOURS_BINS_PER_SEMITONE, MIDI_OFFSET}, postprocessing::helpers::ported::numpy::{global_max, max_3d_for_axis0, min_3d_for_axis0}};

use super::ported::librosa::{hz_to_midi, midi_to_hz};

/// Mutate onsets and frames to have 0s outside of the frequency bounds.
///
/// # Arguments
///
/// * `onsets` - Onsets output from evaluate_model.
/// * `frames` - Frames output from evaluate_model.
/// * `max_freq` - Maximum non-0 frequency in Hz.
/// * `min_freq` - Minimum non-0 frequency in Hz.
pub fn constrain_frequency(
    onsets: &mut [Vec<f32>],
    frames: &mut [Vec<f32>],
    max_freq: Option<f32>,
    min_freq: Option<f32>,
) {
    if let Some(max_freq) = max_freq {
        let max_freq_idx = hz_to_midi(max_freq) as usize - MIDI_OFFSET;
        for onset in onsets.iter_mut() {
            for i in max_freq_idx..onset.len() {
                onset[i] = 0.0;
            }
        }
        for frame in frames.iter_mut() {
            for i in max_freq_idx..frame.len() {
                frame[i] = 0.0;
            }
        }
    }

    if let Some(min_freq) = min_freq {
        let min_freq_idx = hz_to_midi(min_freq) as usize - MIDI_OFFSET;
        for onset in onsets.iter_mut() {
            for i in 0..min_freq_idx {
                onset[i] = 0.0;
            }
        }
        for frame in frames.iter_mut() {
            for i in 0..min_freq_idx {
                frame[i] = 0.0;
            }
        }
    }
}

/// Infer onsets from large changes in frame amplitudes.
///
/// # Arguments
///
/// * `onsets` - Onsets output from evaluate_model.
/// * `frames` - Frames output from evaluate_model.
/// * `n_diff` - Number of differences to compute.
///
/// # Returns
///
/// * A 2D array with the inferred onsets.
pub fn get_inferred_onsets(onsets: &[Vec<f32>], frames: &[Vec<f32>], n_diff: usize) -> Vec<Vec<f32>> {
    let diffs: Vec<Vec<Vec<f32>>> = (1..=n_diff).map(|n| {
        let mut frames_appended = vec![vec![0.0; frames[0].len()]; n];
        frames_appended.extend_from_slice(frames);
        let n_plus = &frames_appended[n..];
        let minus_n = &frames_appended[..frames_appended.len() - n];
        n_plus.iter().zip(minus_n.iter()).map(|(row_plus, row_minus)| {
            row_plus.iter().zip(row_minus.iter()).map(|(v_plus, v_minus)| v_plus - v_minus).collect()
        }).collect()
    }).collect();

    let mut frame_diff = min_3d_for_axis0(&diffs);

    // frame_diff[frame_diff < 0] = 0
    frame_diff.iter_mut().for_each(|row| row.iter_mut().for_each(|v| *v = v.max(0.0)));

    // frame_diff[:n_diff, :] = 0
    for r in 0..n_diff {
        frame_diff[r].fill(0.0);
    }

    // rescale to have the same max as onsets
    let onset_max = global_max(onsets);
    let frame_diff_max = global_max(&frame_diff);
    frame_diff.iter_mut().for_each(|row| row.iter_mut().for_each(|v| *v = (onset_max * *v) / frame_diff_max));

    // use the max of the predicted onsets and the differences
    max_3d_for_axis0(&[onsets.to_vec(), frame_diff])
}

/// Return a symmetric gaussian window.
///
/// The gaussian window is defined as:
///   w(n) = exp(-1/2 * (n / sigma)^2)
///
/// # Arguments
///
/// * `m` - Number of points in the output window. If zero or less, an empty array is returned.
/// * `std` - The standard deviation, sigma.
///
/// # Returns
///
/// * The window, with the maximum value normalized to 1.
pub fn gaussian(m: usize, std: f32) -> Vec<f32> {
    if m == 0 {
        return vec![];
    }

    let midpoint = (m - 1) as f32 / 2.0;
    (0..m).map(|n| {
        
        (-(n as f32 - midpoint).powi(2) / (2.0 * std.powi(2))).exp()
    }).collect()
}

/// Converts a MIDI pitch to a contour bin.
///
/// # Arguments
///
/// * `pitch_midi` - The MIDI pitch.
///
/// # Returns
///
/// * The corresponding contour bin.
pub fn midi_pitch_to_contour_bin(pitch_midi: f32) -> f32 {
    12.0 * CONTOURS_BINS_PER_SEMITONE * (midi_to_hz(pitch_midi) / ANNOTATIONS_BASE_FREQUENCY).log2()
}
