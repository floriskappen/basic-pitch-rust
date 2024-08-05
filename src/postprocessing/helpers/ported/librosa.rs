/* PORTED LIBROSA FUNCTIONS */

use crate::constants::{ANNOT_N_FRAMES, AUDIO_SAMPLE_RATE, FFT_HOP, WINDOW_OFFSET};

/// Converts a frequency in Hz to the corresponding MIDI pitch.
/// 
/// # Arguments
/// 
/// * `hz` - A frequency in Hz.
/// 
/// # Returns
/// 
/// * The corresponding MIDI pitch.
pub fn hz_to_midi(hz: f32) -> f32 {
    12.0 * (hz.log2() - 440.0f32.log2()) + 69.0
}

/// Converts a MIDI pitch to the corresponding frequency in Hz.
/// 
/// # Arguments
/// 
/// * `midi` - A MIDI pitch.
/// 
/// # Returns
/// 
/// * The corresponding frequency in Hz.
pub fn midi_to_hz(midi: f32) -> f32 {
    440.0 * 2.0f32.powf((midi - 69.0) / 12.0)
}

/// Converts from the model's "frame" time to seconds.
/// 
/// # Arguments
/// 
/// * `frame` - The model's "frame".
/// 
/// # Returns
/// 
/// * The time the frame maps to in seconds.
pub fn model_frame_to_time(frame: usize) -> f32 {
    (frame as f32 * FFT_HOP as f32) / AUDIO_SAMPLE_RATE as f32 - WINDOW_OFFSET * (frame as f32 / ANNOT_N_FRAMES as f32).floor()
}
