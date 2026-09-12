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
The public website ID is configured in the Astro layout and standalone SET page; no account password
or API key is included in the website.

The official tracker loads directly in each page head. Shared click handling is
in `public/analytics.js`.
Click events cover email links, CV PDFs, other PDFs, and outbound links, including
links rendered after page load. PDF events measure clicks on the website, not
completed downloads or direct visits to a PDF. Analytics starts collecting after
activation; it does not recover historical visits or identify visitors by name.

Tracking is restricted to the custom domains and GitHub Pages hostname, respects
Do Not Track, and excludes query strings and URL fragments. Event destinations
also omit queries and fragments; email clicks don't include the address. Local
previews do not record visits. Ad blockers and visitor privacy settings can
prevent collection. The tracker never delays or cancels link navigation.

To verify a deployment, visit the live site and confirm a page view and a link
click in your Umami dashboard. Do not put Umami API keys in site code.

### Additional analytics

- Google Search Console: domain property `m-mcmanus.com`, verified using a TXT
  record in Cloudflare DNS. Keep that verification record in place.
- Umami: saved goals for CV page visits and CV PDF clicks; the Journeys view
  shows navigation paths using existing page and event data.
- Microsoft Clarity: project `yha9iyyzfk` (Matt McManus Portfolio), loaded by
  `public/analytics.js` on both Astro pages and the standalone SET game.
  Both Consent V2 storage categories are denied before loading the tracker,
  so Clarity operates in limited, cookieless mode. No cross-page session
  continuity is promised. Do Not Track and Global Privacy Control prevent
  loading Clarity. Its dashboard may take up to two hours to show new data.
  Public disclosure is at `/privacy/`.
