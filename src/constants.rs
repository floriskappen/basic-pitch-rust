// Inference
pub const AUDIO_SAMPLE_RATE: usize = 22050;
pub const FFT_HOP: usize = 256;
pub const ANNOTATIONS_FPS: usize = AUDIO_SAMPLE_RATE / FFT_HOP;
pub const AUDIO_WINDOW_LENGTH: usize = 2;
pub const AUDIO_N_SAMPLES: usize = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP;
pub const MODEL_PATH: &str = "./model/icassp_2022_nmp.onnx";

// MIDI Conversion
pub const MIDI_OFFSET: usize = 21;
