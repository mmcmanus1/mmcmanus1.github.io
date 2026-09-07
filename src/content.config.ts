import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const papers = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/papers' }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    excerpt: z.string(),
    paperUrl: z.string(),
    slidesUrl: z.string().optional(),
    citation: z.string(),
  }),
});

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    description: z.string().optional(),
    tags: z.array(z.string()).optional(),
    draft: z.boolean().default(false),
    order: z.number().optional(),
    unlisted: z.boolean().default(false),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    tagline: z.string(),
    date: z.date(),
    description: z.string(),
    projectUrl: z.string().optional(),
    repoUrl: z.string().optional(),
    tags: z.array(z.string()).optional(),
  }),
});

export const collections = { papers, blog, projects };
