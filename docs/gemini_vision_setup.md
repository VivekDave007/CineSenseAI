# Gemini Vision Setup

The uploaded-image path now uses Gemini 2.5 Flash as the primary multimodal analyzer.

## Required `.env` entries

```env
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

## Runtime behavior

- `Auto` uses Gemini for uploaded image analysis when the Gemini key is configured.
- `Gemini 2.5 Flash` forces Gemini for supported chat/image requests.
- `Local Only` skips external APIs and falls back to the local ResNet50 vision pipeline.
- If Gemini fails during image analysis, CineSense AI falls back to the existing local vision classifier.
