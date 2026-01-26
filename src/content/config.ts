import { defineCollection, z } from 'astro:content';

const papers = defineCollection({
  type: 'content',
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
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.date(),
    description: z.string().optional(),
    tags: z.array(z.string()).optional(),
  }),
});

const projects = defineCollection({
  type: 'content',
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
