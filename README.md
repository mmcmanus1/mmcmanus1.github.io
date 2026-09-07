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

## Analytics

Umami Cloud tracks anonymous page visits, traffic sources, approximate locations,
devices, and sessions. View results at https://cloud.umami.is/ for `m-mcmanus.com`.
The public website ID is configured in `public/analytics.js`; no account password
or API key is included in the website.

The shared tracker is loaded by the Astro layout and the standalone SET game.
Click events cover email links, CV PDFs, other PDFs, and outbound links, including
links rendered after page load. PDF events measure clicks on the website, not
completed downloads or direct visits to a PDF. Analytics starts collecting after
activation; it does not recover historical visits or identify visitors by name.

Tracking is restricted to the custom domains and GitHub Pages hostname, respects
Do Not Track, and excludes query strings and URL fragments. Event destinations
also omit queries and fragments; email clicks don't include the address. Local
previews do not load the tracker. Ad blockers and visitor privacy settings can
prevent collection. The tracker never delays or cancels link navigation.

To verify a deployment, visit the live site and confirm a page view and a link
click in your Umami dashboard. Do not put Umami API keys in site code.
