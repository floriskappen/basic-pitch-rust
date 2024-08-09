use midly::num::u14;
use midly::Format;
use midly::Header;
use midly::MetaMessage;
use midly::MidiMessage;
use midly::PitchBend;
use midly::Smf;
use midly::Timing;
use midly::Track;
use midly::TrackEvent;
use midly::TrackEventKind;
use midly::num::u7;

use std::cmp::max_by;
use std::io::Cursor;

use crate::constants::TICKS_PER_BEAT;

use super::note_event_times::NoteEventTime;

#[derive(Debug, Clone)]
struct TrackEventAbsolute<'a> {
    tick: u32,
    kind: TrackEventKind<'a>,
    is_note_on: Option<bool>
}

pub fn generate_ordered_midi_events(note_events: Vec<NoteEventTime>, ticks_per_second: f64) -> Vec<TrackEvent<'static>> {
    // notes.sort_by(|a, b| b.start_time_seconds.partial_cmp(&a.duration_seconds).unwrap());

    let mut track_events_absolute: Vec<TrackEventAbsolute> = vec![];
    for note_event in note_events {
        // NoteOn event
        let start_tick = (note_event.start_time_seconds as f64 * ticks_per_second).round() as u32;
        let velocity = (note_event.amplitude * 127.0).round() as u8;
        let key = u7::new(note_event.pitch_midi as u8);
        track_events_absolute.push(TrackEventAbsolute {
            tick: start_tick,
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOn {
                    key,
                    vel: u7::new(velocity),
                },
            },
            is_note_on: Some(true)
        });

        // NoteOff event
        let end_tick = (note_event.duration_seconds as f64 * ticks_per_second).round() as u32 + start_tick;
        track_events_absolute.push(TrackEventAbsolute {
            tick: end_tick,
            kind: TrackEventKind::Midi {
                channel: 0.into(),
                message: MidiMessage::NoteOff {
                    key,
                    vel: u7::new(velocity),
                },
            },
            is_note_on: Some(false)
        });

        // Pitch bends
        if let Some(pitch_bends) = note_event.pitch_bends {
            for (i, &pitch_bend) in pitch_bends.iter().enumerate() {
                let mut bend_tick = (((i as f32 * note_event.duration_seconds) / pitch_bends.len() as f32) * ticks_per_second as f32) as u32 + start_tick;
                if i == 0 {
                    bend_tick += 1; // Prevent the bend from starting at the exact same time as the NoteOn event. This would otherwise cause some issues later on when we try to swap NoteOn and NoteOff events to prevent bugs in the MIDI output
                }
                track_events_absolute.push(TrackEventAbsolute {
                    tick: bend_tick,
                    kind: TrackEventKind::Midi {
                        channel: 0.into(),
                        message: MidiMessage::PitchBend {
                            bend: PitchBend((pitch_bend as u16 + 0x2000).into()),
                        },
                    },
                    is_note_on: None
                });
            }
        }
    }

    track_events_absolute.sort_by(|a, b| a.tick.cmp(&b.tick));

    let mut track_events = vec![];

    let mut i = 0;
    while i < track_events_absolute.len() {
        let track_event_absolute = &track_events_absolute[i];
        let mut delta = if i == 0 { track_event_absolute.tick } else {
            track_event_absolute.tick - track_events_absolute[i-1].tick
        };

        if track_event_absolute.is_note_on.is_some_and(|v| v == true) {
            // Check if the next event is a NoteOff and it's on the same tick, if so, we want to add that one first. MIDI doesn't like it when a note is pressed again before it was let go.
            if i + 1 < track_events_absolute.len() && track_events_absolute[i+1].is_note_on.is_some_and(|v| v == false) && track_events_absolute[i+1].tick == track_event_absolute.tick {
                track_events.push(TrackEvent {
                    delta: delta.into(),
                    kind: track_events_absolute[i+1].kind
                });
                delta = 0;
                i += 1;
            }
        }

        track_events.push(TrackEvent {
            delta: delta.into(),
            kind: track_event_absolute.kind
        });

        i += 1;
    }

    return track_events;
}

/// Generate MIDI file data from note events.
///
/// # Arguments
///
/// * `notes` - List of time-based note events.
///
/// # Returns
///
/// * A vector of bytes representing the MIDI file.
pub fn generate_midi_file_data(notes: &[NoteEventTime], beats_per_minute: u32) -> Vec<u8> {
    let timing = Timing::Metrical(TICKS_PER_BEAT.into());
    let ticks_per_second = (TICKS_PER_BEAT as f64) * (beats_per_minute as f64) / 60.0;

    // Initialize the track
    let mut smf = Smf::new(
        Header {
            format: Format::SingleTrack,
            timing
        }
    );
    let mut track = Track::new();

    // Set tempo to match the BPM
    track.push(TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(MetaMessage::Tempo((60_000_000 / beats_per_minute).into()))
    });

    let track_events = generate_ordered_midi_events(notes.to_vec().clone(), ticks_per_second);
    for track_event in track_events {
        track.push(track_event)
    }

    smf.tracks.push(track);

    let mut buffer = Vec::new();
    smf.write_std(&mut Cursor::new(&mut buffer)).unwrap();

    buffer
}
