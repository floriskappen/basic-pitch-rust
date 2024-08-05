use std::{error::Error, fs::File, io::Write, path::Path};

use inference::run_inference;
use postprocessing::{midi::generate_midi_file_data, note_event_frames::{add_pitch_bends_to_note_events, output_to_notes_poly}, note_event_times::note_frames_to_time};

pub mod constants;
pub mod inference;
pub mod preprocessing {
    pub mod load_audio;
    pub mod windowed_audio;
}
pub mod postprocessing {
    pub mod helpers {
        pub mod ported {
            pub mod librosa;
            pub mod numpy;
        }
        pub mod helpers;
    }
    pub mod note_event_frames;
    pub mod note_event_times;
    pub mod midi;
}

fn main() -> Result<(), Box<dyn Error>> {
    let (contours_array, frames_array, onsets_array) = run_inference("test_data/C_major.wav")?;

    let contours: Vec<Vec<f32>> = contours_array.outer_iter().map(|row| row.to_vec()).collect();
    let frames: Vec<Vec<f32>> = frames_array.outer_iter().map(|row| row.to_vec()).collect();
    let onsets: Vec<Vec<f32>> = onsets_array.outer_iter().map(|row| row.to_vec()).collect();

    let note_event_frames = output_to_notes_poly(
        frames,
        onsets,
        0.5,
        0.3,
        5,
        true,
        None,
        None,
        true,
        11,
    );

    let notes_event_frames_with_bend = add_pitch_bends_to_note_events(
        &contours,
        &note_event_frames,
        25
    );

    let note_event_times = note_frames_to_time(&notes_event_frames_with_bend);
    let midi_buffer = generate_midi_file_data(&note_event_times);

    let output_file_path = Path::new("./output/output.midi");
    let mut file = File::create(output_file_path)?;
    file.write_all(&midi_buffer)?;

    // println!("{:?}", note_event_times);

    Ok(())
}
