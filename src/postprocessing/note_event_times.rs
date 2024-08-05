use super::{helpers::ported::librosa::model_frame_to_time, note_event_frames::NoteEventFrame};

#[derive(Debug, Clone)]
pub struct NoteEventTime {
    pub start_time_seconds: f32,
    pub duration_seconds: f32,
    pub pitch_midi: usize,
    pub amplitude: f32,
    pub pitch_bends: Option<Vec<f32>>,
}

/// Convert note frames to time-based note events.
///
/// # Arguments
///
/// * `notes` - List of note events.
///
/// # Returns
///
/// * List of time-based note events.
pub fn note_frames_to_time(notes: &[NoteEventFrame]) -> Vec<NoteEventTime> {
    notes.iter().map(|note| {
        NoteEventTime {
            pitch_midi: note.pitch_midi,
            amplitude: note.amplitude,
            pitch_bends: note.pitch_bends.clone(),
            start_time_seconds: model_frame_to_time(note.start_frame),
            duration_seconds: model_frame_to_time(note.start_frame + note.duration_frames) - model_frame_to_time(note.start_frame),
        }
    }).collect()
}
