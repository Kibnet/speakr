# Documentation screenshots

## Synthetic ASR editor regression tests

The ASR fixture generates tones and invented transcripts. It overrides database,
uploads, providers and queues before importing the app; mutable data lives in
`/tmp`. Use a disposable container without database or upload mounts. No login
credentials are needed. Scripts verify the fixture readiness route and refuse
the normal application/preview ports.

From the repository root, build the branch and start the fixture:

```bash
docker build -t speakr-asr-fixture .
docker run --rm --name speakr-asr-fixture \
  -p 127.0.0.1:8911:8899 --tmpfs /tmp:rw,size=2g \
  -e ASR_SPLIT_SMOKE=1 -e ASR_SEGMENT_SMOKE=1 \
  --entrypoint python speakr-asr-fixture \
  scripts/screenshots/asr-editor-fixture.py
```

In another terminal, install the existing screenshot tooling and run from the root:

```bash
npm --prefix scripts/screenshots install
npm --prefix scripts/screenshots exec -- playwright install chromium
export SPEAKR_URL=http://127.0.0.1:8911
node scripts/screenshots/asr-editor-demo.mjs
node scripts/screenshots/asr-editor-demo.mjs --mobile
node scripts/screenshots/asr-editor-demo.mjs --without-segmenter
SPEAKR_REDUCED_MOTION=1 node scripts/screenshots/asr-workplace-smoke.mjs
node scripts/screenshots/asr-spectrum-zoom-smoke.mjs
node scripts/screenshots/asr-spectrum-zoom-edges.mjs
```

The demo checks playback, zoom boundaries, cached pan/Fit, marker-based splitting
and saved metadata. The compatibility case removes `Intl.Segmenter` before app
bootstrap and verifies editing and saving. Workplace tests cover draft/save races,
speaker tools, recognition proposals, boundary expansion and mobile layout. Zoom
tests also cover long/fractional segments, tiny intervals, end-of-file clipping,
failed Fit retry, cancellation and late lease release.

JSON results, screenshots and finalized WebM videos go under `output/playwright/`.
Review artifacts before publishing. Reduced motion is a browser test setting;
these scenarios do not override deployed styles. For a baseline capture, run the
same fixture helper against an unmodified upstream checkout in a separate
disposable container, then run
`SPEAKR_URL=http://127.0.0.1:8912 node scripts/screenshots/asr-editor-demo.mjs --before`.
The committed examples use synthetic tones, not recorded voices.

Reproducible screenshot harness for the docs (`docs/screenshots.md`, README,
user guide pages). Every shot is defined as code, so after a UI change the
whole gallery can be regenerated identically.

## Standards

- Desktop: 1440x1080 viewport (exact 4:3), device scale 1.
- Mobile: 390x844 viewport at 3x.
- Dark mode by default, with the color scheme varied across shots
  (`blue`, `emerald`, `purple`, `rose`, `amber`, `teal`) to show the themes.
- Fixed locale (en-US) and timezone (America/Chicago); animations frozen.

## Usage

```bash
cd scripts/screenshots
# Node 20+ required (nvm use 22)
npm install && npx playwright install chromium   # once
export SPEAKR_URL=https://your-dev-instance      # defaults to spdev
export SPEAKR_EMAIL=... SPEAKR_PASSWORD=...
# or: export SPEAKR_SESSION_COOKIE='session=<signed session value>'
node capture.mjs --list                          # see all defined shots
node capture.mjs --file=shots/core.mjs           # one area
node capture.mjs --only=main-view,upload-modal   # specific shots
node capture.mjs --out=docs                      # write into docs/ (publish)
```

Without `--out=docs`, images land in `./out/` for review.

## Adding shots

Add an entry to a file in `shots/` (or a new file there):

```js
export default [
  {
    name: 'main-view',                    // output filename (no extension)
    description: 'The main view: recordings list, transcript, and summary',
    theme: { dark: true, scheme: 'blue' },
    // mobile: true,                      // phone viewport instead
    run: async (page) => {                // drive the page into the state
      await go(page, '/');
      await openRecordingByTitle(page, 'Some Recording');
    },
  },
];
```

Helpers in `helpers.mjs`: `go(page, path)`, `openRecordingByTitle(page, title)`,
`clickVisible(page, selector)`, `settle(page)`. Text locators match hidden
template duplicates in this app — always use the visibility-filtering helpers.

Content on screen comes from the dev instance's data; pick recordings whose
titles/content look presentable (see existing captions in docs/screenshots.md
for the intent of each image).
