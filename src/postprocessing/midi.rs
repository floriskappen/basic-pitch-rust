use midly::Header;
use midly::PitchBend;
use midly::Smf;
use midly::Track;
use midly::TrackEventKind;
use midly::num::u7;

use std::io::Cursor;

use super::note_event_times::NoteEventTime;

/// Generate MIDI file data from note events.
///
/// # Arguments
///
/// * `notes` - List of time-based note events.
///
/// # Returns
///
/// * A vector of bytes representing the MIDI file.
pub fn generate_midi_file_data(notes: &[NoteEventTime]) -> Vec<u8> {
    let mut smf = Smf::new(
        Header {
            format: midly::Format::SingleTrack,
            timing: midly::Timing::Metrical(midly::num::u15::new(480))
        }
    );
    let mut track = Track::new();

    for note in notes {
        track.push(midly::TrackEvent {
            delta: 0.into(),
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: midly::MidiMessage::NoteOn {
                    key: u7::new(note.pitch_midi as u8),
                    vel: u7::new((note.amplitude * 127.0) as u8),
                },
            },
        });

        track.push(midly::TrackEvent {
            delta: ((note.duration_seconds * 480.0) as u32).into(),
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: midly::MidiMessage::NoteOff {
                    key: u7::new(note.pitch_midi as u8),
                    vel: u7::new(0),
                },
            },
        });

        if let Some(pitch_bends) = &note.pitch_bends {
            for (i, &bend) in pitch_bends.iter().enumerate() {
                track.push(midly::TrackEvent {
                    delta:( ((i as f32 * note.duration_seconds) / pitch_bends.len() as f32 * 480.0) as u32).into(),
                    kind: TrackEventKind::Midi {
                        channel: 0.into(),
                        message: midly::MidiMessage::PitchBend {
                            bend: PitchBend::from_f64(bend as f64)
                        },
                    },
                });
            }
        }
    }

    smf.tracks.push(track);

    let mut buffer = Vec::new();
    smf.write_std(&mut Cursor::new(&mut buffer)).unwrap();

    buffer
}
