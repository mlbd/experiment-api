# Background Remover API (remove.bg-like, self-hosted)

This API removes background from photos/logos and returns a transparent PNG.

## What makes it better than basic rembg?

- Uses stronger models: BRIA RMBG + BiRefNet fallback
- Auto-detects "logo-like" vs "photo-like" and chooses different refinement
- Upscales alpha to keep original resolution
- Adds halo/fringe cleanup for solid-color backgrounds

> Licensing note:
> BRIA RMBG-1.4 is non-commercial by default. Commercial use requires a BRIA license.

## Endpoints

### Health

GET /health

### Remove background

POST /remove-bg
Form-data: `image` (or `file`) = your upload ( required )

Example:

```bash
curl -s -F image=@input.jpg http://localhost:5000/remove-bg -o output.png
```
