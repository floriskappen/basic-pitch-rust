# basic-pitch-rust

a full implementation of the [basic-pitch](https://github.com/spotify/basic-pitch/) library in rust.

the only other rust implementation I could find was [one by w-ensink](https://github.com/w-ensink/basic_pitch) but that is essentially a port of a c++ implementation from a different project called [NeuralNote](https://github.com/DamRsn/NeuralNote). this project is a full implementation from scratch in rust. i am not at all experienced with working with neural networks or the ndarray crate in rust so i expect things to not be optimal.

## approach
i have taken inspiration mostly from the [python basic-pitch library](https://github.com/spotify/basic-pitch/). but that falls short in some situations because it uses [scipy](https://scipy.org/) and [librosa](https://librosa.org/doc/latest/index.html) functions which are not available in rust. thankfully the spotify team also made a [typescript implementation of the basic-pitch library](https://github.com/spotify/basic-pitch-ts/tree/main) which has already implemented all the necessary functions in plain typescript. i have used this to write rust implementations

## neural network
this implementation uses the provided neural network in onnx format together with the [ort crate](https://crates.io/crates/ort). this seemed like the most cross-platform friendly and simple way to make it work.

## reason
i created this implementation because i wanted it to exist for my own project. but i have tried to set it up as generically as possible so it might also be useful to others. in my own project i am working with audio data that is already in memory and i am trying to use this to detect notes in as real-time as possible so this project may have some additional functionality to make that easier. sadly, as many others have noted in issues of the basic-pitch repositories, the algorithms used by basic-pitch do not allow for real-time AMT.

## what does it not do
- this project does not include any way to train the model from scratch. for that, please refer to the [python implementation](https://github.com/spotify/basic-pitch/)

## contributing
i don't really have experience managing repositories but if you'd like to contribute feel free to.

some things:
- docstings
- it would be nice to implement end to end tests so we can easily verify if stuff is breaking
- ofcourse unit tests would be cool too but yeah idk who would want to write that
- some functions could probably be optimized since i am not that experienced with ndarray/ort
