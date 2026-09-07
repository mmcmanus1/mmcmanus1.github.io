(() => {
  const domains = ['m-mcmanus.com', 'www.m-mcmanus.com', 'mmcmanus1.github.io'];
  if (!domains.includes(window.location.hostname) || navigator.doNotTrack === '1') return;

  const tracker = document.createElement('script');
  tracker.src = 'https://cloud.umami.is/script.js';
  tracker.defer = true;
  tracker.dataset.websiteId = 'cf6d6d80-b65a-441f-a1e0-ad139be602cd';
  tracker.dataset.domains = domains.join(',');
  tracker.dataset.doNotTrack = 'true';
  tracker.dataset.excludeSearch = 'true';
  tracker.dataset.excludeHash = 'true';
  document.head.appendChild(tracker);

  // Delegation also covers links rendered later by the standalone game.
  document.addEventListener('click', (click) => {
    const link = click.target instanceof Element ? click.target.closest('a[href]') : null;
    if (!link || !window.umami) return;

    try {
      const url = new URL(link.href, window.location.href);
      let event;
      const data = {};

      if (url.protocol === 'mailto:') {
        event = 'Email click';
      } else if (url.protocol === 'https:' || url.protocol === 'http:') {
        if (url.origin === window.location.origin && url.pathname === '/files/matt-mcmanus-resume-2026.pdf') {
          event = 'CV PDF click';
        } else if (/\.pdf$/i.test(url.pathname)) {
          event = 'PDF click';
        } else if (url.origin !== window.location.origin) {
          event = 'Outbound link';
        }
        if (event) data.destination = url.origin + url.pathname;
      }

      // Umami uses keepalive delivery; navigation remains immediate.
      if (event) Promise.resolve(window.umami.track(event, data)).catch(() => {});
    } catch {
      // Malformed links or a blocked tracker must not affect the website.
    }
  });
})();
