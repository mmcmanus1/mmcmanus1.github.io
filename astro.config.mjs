import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://mmcmanus1.github.io',
  output: 'static',
  build: {
    format: 'directory'
  }
});
