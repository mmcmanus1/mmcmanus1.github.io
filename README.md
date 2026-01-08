# Matt McManus Personal Website

A minimal, fast personal website built with [Astro](https://astro.build).

## Tech Stack

- **Framework**: Astro 4.x (compiles to static HTML)
- **Styling**: Vanilla CSS with CSS custom properties
- **Content**: Astro Content Collections
- **Hosting**: GitHub Pages

## Development

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
src/
├── components/   # Astro components (Header, Nav, Footer)
├── content/      # Content collections (papers, blog)
├── layouts/      # Page layouts
├── pages/        # Routes
└── styles/       # Global CSS
public/
├── files/        # PDFs (papers, CV)
└── profile.png   # Profile photo
```

## Deployment

Push to `main` branch to trigger GitHub Pages deployment via GitHub Actions.
