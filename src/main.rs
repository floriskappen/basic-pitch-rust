use std::error::Error;

use inference::run_inference;

mod constants;
mod inference;
mod preprocessing {
    pub mod load_audio;
    pub mod windowed_audio;
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = run_inference("/Users/kade/git/personal/guitar-gaming/python-experiment/input/single.wav")?;
    println!("{:?}", output);

    Ok(())
}
