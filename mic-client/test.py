import wave, struct, numpy as np, textwrap, pathlib

fname = "startup_test_norm.wav"          # or startup_test.wav
n_head = 64                              # how many samples to show

with wave.open(fname, "rb") as wf:
    nframes = wf.getnframes()
    raw = wf.readframes(nframes)

samples = np.frombuffer(raw, dtype="<i2")      # little-endian int16
print(f"{fname}:  {len(samples)} samples,  sr = {wf.getframerate()} Hz")

print("\nfirst 64 samples:")
print(textwrap.fill(" ".join(map(str, samples[:n_head])), width=80))

print("\nmin =", samples.min(), " max =", samples.max())
print("RMS ~", int(np.sqrt(np.mean(samples.astype(np.float32)**2))))