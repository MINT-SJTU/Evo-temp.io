# Evo-RL Project Website

This is a static project page for open-source release.

## Run locally

```bash
cd website
python -m http.server 8000
# open http://localhost:8000
```

## Replace placeholders

- Hero video: `assets/placeholders/hero_demo.mp4`
- Demo slots:
  - `assets/placeholders/demo_01.mp4`
  - `assets/placeholders/demo_02.mp4`
  - `assets/placeholders/demo_03.mp4`
- Results chart image: `assets/placeholders/results_chart_placeholder.svg`

You can keep the same file names and overwrite files in place.

## Deploy options

- GitHub Pages (recommended): publish the `website/` directory.
- Any static host (Vercel/Netlify/Nginx): serve `website/index.html` as root.
