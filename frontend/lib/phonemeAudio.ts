// Audio files are pre-downloaded to Supabase Storage by
// fastapi-backend/scripts/download_audio.py.
// Original source: Wikimedia Commons (CC BY-SA 3.0 / CC BY-SA 4.0 per file).
// 8 diphthongs (/eɪ/ /aɪ/ /ɔɪ/ /aʊ/ /əʊ/ /ɪə/ /eə/ /ʊə/) have no standalone
// IPA Commons recording and are intentionally omitted — callers receive null
// and should gracefully disable the play button for those phonemes.
const _SUPABASE_URL = (process.env.EXPO_PUBLIC_SUPABASE_URL ?? '').replace(/\/$/, '');
const _STORAGE_BASE = _SUPABASE_URL
  ? `${_SUPABASE_URL}/storage/v1/object/public/pronunciation-audio/phonemes/`
  : null;

// Values are the original Wikimedia filenames, reused as storage path keys.
const IPA_COMMONS_FILE: Record<string, string> = {
  // Monophthong vowels
  '/iː/': 'Close_front_unrounded_vowel.ogg',
  '/ɪ/':  'Near-close_near-front_unrounded_vowel.ogg',
  '/e/':  'Close-mid_front_unrounded_vowel.ogg',
  '/æ/':  'Near-open_front_unrounded_vowel.ogg',
  '/ɑː/': 'Open_back_unrounded_vowel.ogg',
  '/ɒ/':  'Open_back_rounded_vowel.ogg',
  '/ɔː/': 'Open-mid_back_rounded_vowel.ogg',
  '/ʊ/':  'Near-close_near-back_rounded_vowel.ogg',
  '/uː/': 'Close_back_rounded_vowel.ogg',
  '/ʌ/':  'Open-mid_back_unrounded_vowel.ogg',
  '/ɜː/': 'Open-mid_central_unrounded_vowel.ogg',
  '/ə/':  'Mid-central_vowel.ogg',
  // Consonants — stops
  '/p/':  'Voiceless_bilabial_plosive.ogg',
  '/b/':  'Voiced_bilabial_plosive.ogg',
  '/t/':  'Voiceless_alveolar_plosive.ogg',
  '/d/':  'Voiced_alveolar_plosive.ogg',
  '/k/':  'Voiceless_velar_plosive.ogg',
  '/ɡ/':  'Voiced_velar_plosive.ogg',
  // Consonants — fricatives
  '/f/':  'Voiceless_labiodental_fricative.ogg',
  '/v/':  'Voiced_labiodental_fricative.ogg',
  '/θ/':  'Voiceless_dental_fricative.ogg',
  '/ð/':  'Voiced_dental_fricative.ogg',
  '/s/':  'Voiceless_alveolar_sibilant.ogg',
  '/z/':  'Voiced_alveolar_sibilant.ogg',
  '/ʃ/':  'Voiceless_palato-alveolar_sibilant.ogg',
  '/ʒ/':  'Voiced_palato-alveolar_sibilant.ogg',
  '/h/':  'Voiceless_glottal_fricative.ogg',
  // Consonants — affricates
  '/tʃ/': 'Voiceless_palato-alveolar_affricate.ogg',
  '/dʒ/': 'Voiced_palato-alveolar_affricate.ogg',
  // Consonants — nasals
  '/m/':  'Bilabial_nasal.ogg',
  '/n/':  'Alveolar_nasal.ogg',
  '/ŋ/':  'Velar_nasal.ogg',
  // Consonants — approximants
  '/l/':  'Alveolar_lateral_approximant.ogg',
  '/r/':  'Alveolar_approximant.ogg',
  '/j/':  'Palatal_approximant.ogg',
  '/w/':  'Voiced_labio-velar_approximant.ogg',
};

export function phonemeAudioUrl(phoneme: string): string | null {
  if (!_STORAGE_BASE) return null;
  const filename = IPA_COMMONS_FILE[phoneme];
  return filename ? _STORAGE_BASE + filename : null;
}
