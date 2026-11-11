# strumnet

Guitar string and tuning-state classifier that doesn't fall for octave errors.

Tuner apps take the loudest peak in the spectrum and assume it's the fundamental. Phone mics roll off the low end, so on the low strings that peak is often the 2nd harmonic and the tuner reports the wrong string, way out of tune. The plan: render a lot of realistic notes, extract a handful of spectral features and train a small classifier that gets string and tuning state right even when the fundamental is weak.

## Generating the dataset

```bash
pip install -e .
strumnet generate --out data/guitar_notes.csv
```

It prints the split sizes and how a plain peak-picking tuner does on the data, broken down by what made each example hard (weak fundamental, sympathetic resonance, hum).
