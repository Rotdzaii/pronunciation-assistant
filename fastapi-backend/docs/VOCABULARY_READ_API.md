# Vocabulary Read API

Vocabulary is a pronunciation-support feature. These endpoints expose active
seeded pronunciation vocabulary and public active practice sets only.

## Auth Behavior

All vocabulary endpoints require a Supabase user access token:

```text
Authorization: Bearer <supabase-access-token>
```

The API uses the Supabase anon key with the caller token attached so DB1 RLS
policies evaluate the request as `authenticated`. It does not use the service
role client for these read endpoints.

## Endpoints

### GET /vocabulary/items

Lists active vocabulary items.

Query parameters:

- `topic` optional string
- `level` optional string
- `limit` optional integer, default `50`, max `100`
- `offset` optional integer, default `0`, min `0`

Example:

```powershell
curl.exe -H "Authorization: Bearer <supabase-access-token>" `
  "http://localhost:8000/vocabulary/items?topic=final_consonants&limit=20&offset=0"
```

Response shape:

```json
{
  "items": [
    {
      "id": "00000000-0000-0000-0000-000000000000",
      "word": "example",
      "phonetic": "/ig-ZAM-pul/",
      "meaning_vi": "vi du",
      "topic": "final_consonants",
      "level": "A2",
      "difficulty": 2,
      "sample_sentence": "This is an example.",
      "target_phonemes": ["l"],
      "common_mistake_tags": ["final_consonant"],
      "stress_pattern": "2"
    }
  ],
  "limit": 20,
  "offset": 0
}
```

### GET /vocabulary/sets

Lists active public vocabulary sets.

Query parameters:

- `topic` optional string
- `level` optional string
- `limit` optional integer, default `50`, max `100`
- `offset` optional integer, default `0`, min `0`

Example:

```powershell
curl.exe -H "Authorization: Bearer <supabase-access-token>" `
  "http://localhost:8000/vocabulary/sets?limit=10&offset=0"
```

Response shape:

```json
{
  "items": [
    {
      "id": "00000000-0000-0000-0000-000000000000",
      "title": "Final Consonants Practice",
      "description": "Practice words for final consonants.",
      "topic": "final_consonants",
      "level": "A2",
      "is_public": true,
      "is_active": true,
      "item_count": 12
    }
  ],
  "limit": 10,
  "offset": 0
}
```

### GET /vocabulary/sets/{set_id}

Gets one active public set and its active items ordered by `sort_order`.

Example:

```powershell
curl.exe -H "Authorization: Bearer <supabase-access-token>" `
  "http://localhost:8000/vocabulary/sets/<set-id>"
```

Response shape:

```json
{
  "id": "00000000-0000-0000-0000-000000000000",
  "title": "Final Consonants Practice",
  "description": "Practice words for final consonants.",
  "topic": "final_consonants",
  "level": "A2",
  "is_public": true,
  "is_active": true,
  "items": [
    {
      "id": "11111111-1111-1111-1111-111111111111",
      "word": "example",
      "phonetic": "/ig-ZAM-pul/",
      "meaning_vi": "vi du",
      "topic": "final_consonants",
      "level": "A2",
      "difficulty": 2,
      "sample_sentence": "This is an example.",
      "target_phonemes": ["l"],
      "common_mistake_tags": ["final_consonant"],
      "stress_pattern": "2"
    }
  ]
}
```

## Known Limitations

- Read-only API only.
- No vocabulary write endpoints.
- No teacher assignments.
- No vocabulary practice history.
- No frontend integration yet.
- `item_count` is counted from visible set-item links for the returned page.

## Manual Test Checklist

1. Start the backend:

```powershell
cd fastapi-backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. Call each endpoint with a valid Supabase user access token.
3. Confirm `/vocabulary/items` returns active seeded items only.
4. Confirm `/vocabulary/sets` returns the 5 active public seed sets.
5. Confirm `/vocabulary/sets/{set_id}` returns items in set order.
6. Confirm requests without `Authorization` return `401`.
7. Confirm `limit=101` and negative `offset` return validation errors.

## Next Steps

- Add frontend vocabulary screens.
- Add assignment APIs after teacher/class ownership rules are designed.
- Add vocabulary practice history only after the product flow is confirmed.
