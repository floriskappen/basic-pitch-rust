// Inference
pub const AUDIO_SAMPLE_RATE: usize = 22050;
pub const FFT_HOP: usize = 256;
pub const ANNOTATIONS_FPS: usize = AUDIO_SAMPLE_RATE / FFT_HOP; // Should Math.floor
pub const AUDIO_WINDOW_LENGTH: usize = 2;
pub const AUDIO_N_SAMPLES: usize = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP;
pub const MODEL_PATH: &str = "./model/icassp_2022_nmp.onnx";

// MIDI Conversion
pub const MIDI_OFFSET: usize = 21;
pub const ANNOT_N_FRAMES: usize = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH;
pub const WINDOW_OFFSET: f32 =
    ((FFT_HOP / AUDIO_SAMPLE_RATE) * (ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP)) as f32 +
    0.0018; //  this is a magic number, but it's needed for this to align properly
pub const MAX_FREQ_IDX: usize = 87;
pub const CONTOURS_BINS_PER_SEMITONE: f32 = 3.0;
pub const ANNOTATIONS_BASE_FREQUENCY: f32 = 27.5; // lowest key on a piano
pub const ANNOTATIONS_N_SEMITONES: f32 = 88.0; // number of piano keys
pub const N_FREQ_BINS_CONTOURS: usize =
  (ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE) as usize;
