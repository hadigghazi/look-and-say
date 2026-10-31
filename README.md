# Look-and-Say

A browser AAC app for Lebanese children and adults whose speech is hard to understand. The user points at a picture
symbol, by looking at it or touching it, and says its word; the app combines the two clues and speaks the word aloud.
Final project for Natural and Multimodal Interaction, M2 Data Science, Lebanese University.

## What is here so far

- **Grid** (`#/`): the 34 starter words with ARASAAC pictograms, colour-coded by word type, with Lebanese, MSA or English
  labels. Touch a cell to speak it.
- **Check 1** (`#/gaze`): MediaPipe face landmarks in the browser, frame rate and iris position per eye.
- **Check 2** (`#/mic`): microphone capture and automatic detection of single spoken words.
- **Check 3** (`#/templates`): record five words three times each, then test recognition from your own recordings
  (MFCC features + dynamic time warping), with a confusion matrix.

Vocabulary: `data/vocabulary.json`. Audio maths: `src/audio/` (unit tests in `audio.test.ts`).

## Build and deploy

Nothing is built locally. On every push to `main`, GitHub Actions installs, runs the tests and builds the site. When the
repository variable `DEPLOY_ENABLED` is `true`, it also copies the site to the VM, where `deploy/docker-compose.yml`
serves it behind Caddy at https://aac.hadighazi.com (HTTPS lets tablets use the camera and microphone).

## Credits

Pictographic symbols: Sergio Palao. Origin: ARASAAC (http://www.arasaac.org). License: CC (BY-NC-SA). Owner: Government
of Aragon (Spain).
