use crate::constants::{MAX_FREQ_IDX, MIDI_OFFSET, N_FREQ_BINS_CONTOURS};

use super::helpers::{helpers::{constrain_frequency, gaussian, get_inferred_onsets, midi_pitch_to_contour_bin}, ported::numpy::{arg_max, arg_max_axis1, arg_rel_max, global_max, mean_std_dev, where_greater_than_axis1}};

#[derive(Debug, Clone)]
pub struct NoteEventFrame {
    pub start_frame: usize,
    pub duration_frames: usize,
    pub pitch_midi: usize,
    pub amplitude: f32,
    pub pitch_bends: Option<Vec<f32>>,
}

/// Decode raw model output to polyphonic note events.
///
/// # Arguments
///
/// * `frames` - Frame activation matrix (n_times, n_freqs).
/// * `onsets` - Onset activation matrix (n_times, n_freqs).
/// * `onset_thresh` - Minimum amplitude of an onset activation to be considered an onset.
/// * `frame_thresh` - Minimum amplitude of a frame activation for a note to remain "on".
/// * `min_note_len` - Minimum allowed note length in frames.
/// * `infer_onsets` - If true, add additional onsets when there are large differences in frame amplitudes.
/// * `max_freq` - Maximum allowed output frequency, in Hz.
/// * `min_freq` - Minimum allowed output frequency, in Hz.
/// * `melodia_trick` - Remove semitones near a peak.
/// * `energy_tolerance` - Number of frames allowed to drop below 0.
///
/// # Returns
///
/// * A list of tuples [(start_time_seconds, duration_seconds, pitch_midi, amplitude)] where amplitude is a number between 0 and 1.
pub fn output_to_notes_poly(
    mut frames: Vec<Vec<f32>>,
    mut onsets: Vec<Vec<f32>>,
    onset_thresh: f32,
    frame_thresh: f32,
    min_note_len: usize,
    infer_onsets: bool,
    max_freq: Option<f32>,
    min_freq: Option<f32>,
    melodia_trick: bool,
    energy_tolerance: usize,
) -> Vec<NoteEventFrame> {
    let mut inferred_frame_thresh = frame_thresh;
    if inferred_frame_thresh.is_nan() {
        let (mean, std) = mean_std_dev(&frames);
        inferred_frame_thresh = mean + std;
    }

    let n_frames = frames.len();

    // Modifies onsets and frames in place.
    constrain_frequency(&mut onsets, &mut frames, max_freq, min_freq);

    let mut inferred_onsets = onsets.to_vec();
    if infer_onsets {
        inferred_onsets = get_inferred_onsets(&mut onsets, &mut frames, 2);
    }

    // a hacky form of zeros-like
    let mut peak_threshold_matrix = inferred_onsets.iter().map(|o| vec![0.0; o.len()]).collect::<Vec<_>>();
    
    for (row, col) in arg_rel_max(&inferred_onsets, 2) {
        peak_threshold_matrix[row][col] = inferred_onsets[row][col];
    }

    let (mut note_starts, mut freq_idxs) = where_greater_than_axis1(&peak_threshold_matrix, onset_thresh);

    note_starts.reverse();
    freq_idxs.reverse();

    // Deep copy to remaining energy
    let mut remaining_energy = frames.to_vec();

    let mut note_events: Vec<NoteEventFrame> = note_starts
        .iter()
        .zip(freq_idxs.iter())
        .filter_map(|(&note_start_idx, &freq_idx)| {
            // if we're too close to the end of the audio, continue
            if note_start_idx >= n_frames - 1 {
                return None;
            }

            // find time index at this frequency band where the frames drop below an energy threshold
            let mut i = note_start_idx + 1;
            let mut k = 0; // number of frames since energy dropped below threshold
            while i < n_frames - 1 && k < energy_tolerance {
                if remaining_energy[i][freq_idx] < inferred_frame_thresh {
                    k += 1;
                } else {
                    k = 0;
                }
                i += 1;
            }

            i -= k; // go back to frame above threshold

            // if the note is too short, skip it
            if i - note_start_idx <= min_note_len {
                return None;
            }

            for j in note_start_idx..i {
                remaining_energy[j][freq_idx] = 0.0;
                if freq_idx < MAX_FREQ_IDX {
                    remaining_energy[j][freq_idx + 1] = 0.0;
                }
                if freq_idx > 0 {
                    remaining_energy[j][freq_idx - 1] = 0.0;
                }
            }

            // add the note
            let amplitude = frames[note_start_idx..i]
                .iter()
                .map(|row| row[freq_idx])
                .sum::<f32>() / (i - note_start_idx) as f32;

            Some(NoteEventFrame {
                start_frame: note_start_idx,
                duration_frames: i - note_start_idx,
                pitch_midi: freq_idx + MIDI_OFFSET,
                amplitude,
                pitch_bends: None,
            })
        })
        .collect();

    if melodia_trick {
        while global_max(&remaining_energy) > inferred_frame_thresh {
            // We want the (row, column) with the largest value in remaining_energy
            let (i_mid, freq_idx) = remaining_energy.iter().enumerate().fold((0, 0), |(max_row, max_col), (row_idx, row)| {
                let col_max_idx = arg_max(row).unwrap();
                if row[col_max_idx] > remaining_energy[max_row][max_col] {
                    (row_idx, col_max_idx)
                } else {
                    (max_row, max_col)
                }
            });

            remaining_energy[i_mid][freq_idx] = 0.0;
            // forward pass
            let mut i = i_mid + 1;
            let mut k = 0;
            while i < n_frames - 1 && k < energy_tolerance {
                if remaining_energy[i][freq_idx] < inferred_frame_thresh {
                    k += 1;
                } else {
                    k = 0;
                }

                remaining_energy[i][freq_idx] = 0.0;
                if freq_idx < MAX_FREQ_IDX {
                    remaining_energy[i][freq_idx + 1] = 0.0;
                }
                if freq_idx > 0 {
                    remaining_energy[i][freq_idx - 1] = 0.0;
                }

                i += 1;
            }
            let i_end = i - 1 - k;

            // backwards pass
            i = i_mid - 1;
            k = 0;
            while i > 0 && k < energy_tolerance {
                if remaining_energy[i][freq_idx] < inferred_frame_thresh {
                    k += 1;
                } else {
                    k = 0;
                }

                remaining_energy[i][freq_idx] = 0.0;
                if freq_idx < MAX_FREQ_IDX {
                    remaining_energy[i][freq_idx + 1] = 0.0;
                }
                if freq_idx > 0 {
                    remaining_energy[i][freq_idx - 1] = 0.0;
                }

                i -= 1;
            }
            let i_start = i + 1 + k;

            if i_end >= n_frames {
                panic!("i_end is past end of times. (i_end, times.length): ({}, {})", i_end, n_frames);
            }

            // amplitude = np.mean(frames[i_start:i_end, freq_idx])
            let amplitude = frames[i_start..i_end]
                .iter()
                .map(|row| row[freq_idx])
                .sum::<f32>() / (i_end - i_start) as f32;

            if i_end - i_start <= min_note_len {
                // note is too short or too quiet, skip it and remove the energy
                continue;
            }

            // add the note
            note_events.push(NoteEventFrame {
                start_frame: i_start,
                duration_frames: i_end - i_start,
                pitch_midi: freq_idx + MIDI_OFFSET,
                amplitude,
                pitch_bends: None,
            });
        }
    }

    note_events
}

/// Add pitch bends to note events based on the contours.
///
/// # Arguments
///
/// * `contours` - Contours array.
/// * `notes` - List of note events.
/// * `n_bins_tolerance` - Number of bins tolerance.
///
/// # Returns
///
/// * List of note events with pitch bends added.
pub fn add_pitch_bends_to_note_events(
    contours: &[Vec<f32>],
    notes: &[NoteEventFrame],
    n_bins_tolerance: usize,
) -> Vec<NoteEventFrame> {
    let window_length = n_bins_tolerance * 2 + 1;
    let freq_gaussian = gaussian(window_length, 5.0);

    notes.iter().map(|note| {
        let freq_idx = midi_pitch_to_contour_bin(note.pitch_midi as f32).round() as usize;
        let freq_start_idx = freq_idx.saturating_sub(n_bins_tolerance);
        let freq_end_idx = (freq_idx + n_bins_tolerance + 1).min(N_FREQ_BINS_CONTOURS);

        let freq_gaussian_submatrix = &freq_gaussian[
            n_bins_tolerance.saturating_sub(freq_idx)..window_length - (freq_idx.saturating_sub(N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1))
        ];

        let pitch_bend_submatrix: Vec<Vec<f32>> = contours[note.start_frame..note.start_frame + note.duration_frames]
            .iter()
            .map(|d| {
                d[freq_start_idx..freq_end_idx]
                    .iter()
                    .zip(freq_gaussian_submatrix.iter())
                    .map(|(&v, &g)| v * g)
                    .collect()
            })
            .collect();

        let pb_shift = n_bins_tolerance.saturating_sub(freq_idx.saturating_sub(n_bins_tolerance));
        let bends: Vec<isize> = arg_max_axis1(&pitch_bend_submatrix)
            .iter()
            .filter_map(|&v| v.map(|v| v as isize - pb_shift as isize))
            .collect();

        NoteEventFrame {
            start_frame: note.start_frame,
            duration_frames: note.duration_frames,
            pitch_midi: note.pitch_midi,
            amplitude: note.amplitude,
            pitch_bends: Some(bends.iter().map(|&v| v as f32).collect()),
        }
    }).collect()
}