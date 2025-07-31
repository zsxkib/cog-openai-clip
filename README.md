---
name: clip
description: Turn any text or image into a 768-dimensional vector that captures its meaning
github_url: https://github.com/openai/CLIP
paper_url: https://arxiv.org/abs/2103.00020
license_url: https://github.com/openai/CLIP/blob/main/LICENSE
---

[![Replicate](https://replicate.com/openai/clip/badge)](https://replicate.com/openai/clip)

# CLIP

OpenAI's CLIP understands both text and images. Give it either one, and you get back a 768-dimensional vector that captures what it means.

## What you can do with CLIP

CLIP turns text and images into vectors. Since the vectors are in the same space, you can compare them to find similarities across different types of content.

**🔍 Search by meaning, not keywords.** Upload a photo of a red sports car, and CLIP can find text descriptions that match it, like "fast red vehicle" or "crimson automobile."

**🛍️ Build recommendation systems.** Get vectors for your product images and customer searches, then find the closest matches.

**📁 Organize content automatically.** Group similar images together or tag them based on text descriptions.

**🔎 Create multimodal search engines.** Let people search your image library using natural language.

## How CLIP works

CLIP learned to understand both text and images by looking at millions of image-caption pairs. It maps both types of content into the same 768-dimensional space, where similar concepts end up close together.

When you give CLIP text like "a dog playing in a park," it returns a vector. When you give it an image of a dog playing in a park, you get a similar vector. You can then compare these vectors to measure how similar the content is.

## Why this version loads fast ⚡

Most CLIP implementations take 2+ minutes to start because they download a 3.4GB model every time. We store the model weights in a Google Cloud bucket and download them in parallel, so the model loads in about 12 seconds.

Once it's loaded, predictions are instant.

## What makes good results

**For images:** Clear, well-lit photos work best. The model was trained mostly on photographs, so drawings or very stylized images might not work as well.

**For text:** Descriptive phrases work better than single words. Instead of "car," try "red sports car driving down a highway."

**For comparisons:** Compare similar types of content. Product photos work well with product descriptions, but mixing very different domains might give unexpected results.

## When to use CLIP ✅

CLIP works well for:

- **E-commerce search** where people describe what they want instead of using exact product names
- **Content moderation** to find images or text that match certain concepts
- **Media organization** to automatically tag and group visual content
- **Recommendation systems** that need to understand both what people say and what they see

CLIP doesn't work as well for:

- **Text that isn't in English** (it was trained mostly on English)
- **Very abstract or artistic images** that look very different from photographs  
- **Precise object detection** (it understands concepts better than exact locations)

## Research background

CLIP comes from OpenAI's research on contrastive learning. Instead of training separate models for text and images, they trained one model to understand both by showing it millions of images paired with their descriptions.

The key insight was that you don't need carefully labeled datasets. You can just use images and captions that already exist on the internet, then train the model to match them up.

Read the original research: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## What you get back

CLIP returns a single array of 768 numbers (called a vector or embedding) that represents the meaning of your input. These numbers don't mean anything on their own, but you can compare vectors using cosine similarity to measure how similar two pieces of content are.

You can also store these vectors in a vector database to build fast search systems.

## Input requirements

Send either text or an image, not both. If you send both, CLIP will only process the image.

Text can be any length, but shorter, descriptive phrases usually work better than very long passages.

Images can be any common format (JPEG, PNG, etc.). Larger images take longer to process, so resize them if speed matters.

## Licensing 📄

OpenAI released CLIP under the MIT License, which means you can use it for commercial projects. The model is free to use and modify.

---

Built by OpenAI • [Paper](https://arxiv.org/abs/2103.00020) • [GitHub](https://github.com/openai/CLIP)
