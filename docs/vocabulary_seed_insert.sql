-- DB1 vocabulary seed insertion draft.
-- Reviewed SQL draft only. Do not apply automatically.
-- Intended manual application path: Supabase SQL Editor after review.

begin;

with seed_items_raw as (
  select *
  from jsonb_to_recordset($seed_items$
[
  {
    "word": "book",
    "phonetic": "/bÊŠk/",
    "meaning_vi": "sach",
    "topic": "school",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "I put the book on the desk.",
    "target_phonemes": ["b", "ÊŠ", "k"],
    "common_mistake_tags": ["final_stop_confusion", "vowel_length_confusion"],
    "stress_pattern": "BOOK"
  },
  {
    "word": "desk",
    "phonetic": "/desk/",
    "meaning_vi": "ban hoc",
    "topic": "school",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "The desk is near the window.",
    "target_phonemes": ["d", "e", "s", "k"],
    "common_mistake_tags": ["final_consonant", "consonant_cluster_reduction"],
    "stress_pattern": "DESK"
  },
  {
    "word": "class",
    "phonetic": "/klÃ¦s/",
    "meaning_vi": "lop hoc",
    "topic": "school",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "Our class starts at eight.",
    "target_phonemes": ["k", "l", "Ã¦", "s"],
    "common_mistake_tags": ["consonant_cluster_reduction", "ae_e_confusion", "drop_final_consonant"],
    "stress_pattern": "CLASS"
  },
  {
    "word": "student",
    "phonetic": "/ËˆstjuËdÉ™nt/",
    "meaning_vi": "hoc sinh",
    "topic": "school",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "Each student reads aloud.",
    "target_phonemes": ["s", "t", "j", "uË", "d", "É™", "n", "t"],
    "common_mistake_tags": ["word_stress_error", "vowel_length_confusion", "drop_final_consonant"],
    "stress_pattern": "STU-dent"
  },
  {
    "word": "teacher",
    "phonetic": "/ËˆtiËtÊƒÉ™r/",
    "meaning_vi": "giao vien",
    "topic": "school",
    "level": "beginner",
    "difficulty": "medium",
    "sample_sentence": "The teacher gives clear feedback.",
    "target_phonemes": ["t", "iË", "tÊƒ", "É™", "r"],
    "common_mistake_tags": ["short_long_i_confusion", "sh_ch_confusion", "r_l_confusion"],
    "stress_pattern": "TEA-cher"
  },
  {
    "word": "lesson",
    "phonetic": "/ËˆlesÉ™n/",
    "meaning_vi": "bai hoc",
    "topic": "school",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "The lesson is short today.",
    "target_phonemes": ["l", "e", "s", "É™", "n"],
    "common_mistake_tags": ["r_l_confusion", "ae_e_confusion", "word_stress_error"],
    "stress_pattern": "LES-son"
  },
  {
    "word": "family",
    "phonetic": "/ËˆfÃ¦mÉ™li/",
    "meaning_vi": "gia dinh",
    "topic": "daily life",
    "level": "beginner",
    "difficulty": "medium",
    "sample_sentence": "My family eats dinner together.",
    "target_phonemes": ["f", "Ã¦", "m", "É™", "l", "i"],
    "common_mistake_tags": ["ae_e_confusion", "r_l_confusion", "word_stress_error"],
    "stress_pattern": "FAM-i-ly"
  },
  {
    "word": "breakfast",
    "phonetic": "/ËˆbrekfÉ™st/",
    "meaning_vi": "bua sang",
    "topic": "daily life",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "I eat breakfast at seven.",
    "target_phonemes": ["b", "r", "e", "k", "f", "É™", "s", "t"],
    "common_mistake_tags": ["consonant_cluster_reduction", "word_stress_error", "drop_final_consonant"],
    "stress_pattern": "BREAK-fast"
  },
  {
    "word": "kitchen",
    "phonetic": "/ËˆkÉªtÊƒÉªn/",
    "meaning_vi": "nha bep",
    "topic": "daily life",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The kitchen is clean.",
    "target_phonemes": ["k", "Éª", "tÊƒ", "Éª", "n"],
    "common_mistake_tags": ["sh_ch_confusion", "short_long_i_confusion", "word_stress_error"],
    "stress_pattern": "KITCH-en"
  },
  {
    "word": "rice",
    "phonetic": "/raÉªs/",
    "meaning_vi": "com",
    "topic": "daily life",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "We have rice for lunch.",
    "target_phonemes": ["r", "aÉª", "s"],
    "common_mistake_tags": ["r_l_confusion", "drop_final_consonant"],
    "stress_pattern": "RICE"
  },
  {
    "word": "milk",
    "phonetic": "/mÉªlk/",
    "meaning_vi": "sua",
    "topic": "daily life",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "The child drinks milk every day.",
    "target_phonemes": ["m", "Éª", "l", "k"],
    "common_mistake_tags": ["r_l_confusion", "short_long_i_confusion", "final_consonant"],
    "stress_pattern": "MILK"
  },
  {
    "word": "phone",
    "phonetic": "/fÉ™ÊŠn/",
    "meaning_vi": "dien thoai",
    "topic": "technology",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "My phone is on the table.",
    "target_phonemes": ["f", "É™ÊŠ", "n"],
    "common_mistake_tags": ["vowel_length_confusion", "drop_final_consonant"],
    "stress_pattern": "PHONE"
  },
  {
    "word": "screen",
    "phonetic": "/skriËn/",
    "meaning_vi": "man hinh",
    "topic": "technology",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The screen is too bright.",
    "target_phonemes": ["s", "k", "r", "iË", "n"],
    "common_mistake_tags": ["consonant_cluster_reduction", "short_long_i_confusion", "r_l_confusion"],
    "stress_pattern": "SCREEN"
  },
  {
    "word": "click",
    "phonetic": "/klÉªk/",
    "meaning_vi": "nhan chuot",
    "topic": "technology",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "Click the blue button.",
    "target_phonemes": ["k", "l", "Éª", "k"],
    "common_mistake_tags": ["consonant_cluster_reduction", "short_long_i_confusion", "r_l_confusion"],
    "stress_pattern": "CLICK"
  },
  {
    "word": "chat",
    "phonetic": "/tÊƒÃ¦t/",
    "meaning_vi": "tro chuyen",
    "topic": "technology",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "We chat after class.",
    "target_phonemes": ["tÊƒ", "Ã¦", "t"],
    "common_mistake_tags": ["sh_ch_confusion", "ae_e_confusion", "drop_final_consonant"],
    "stress_pattern": "CHAT"
  },
  {
    "word": "video",
    "phonetic": "/ËˆvÉªdiÉ™ÊŠ/",
    "meaning_vi": "video",
    "topic": "technology",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The video is very clear.",
    "target_phonemes": ["v", "Éª", "d", "i", "É™ÊŠ"],
    "common_mistake_tags": ["short_long_i_confusion", "word_stress_error"],
    "stress_pattern": "VID-e-o"
  },
  {
    "word": "email",
    "phonetic": "/ËˆiËmeÉªl/",
    "meaning_vi": "thu dien tu",
    "topic": "work",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "Please send an email today.",
    "target_phonemes": ["iË", "m", "eÉª", "l"],
    "common_mistake_tags": ["short_long_i_confusion", "r_l_confusion", "word_stress_error"],
    "stress_pattern": "E-mail"
  },
  {
    "word": "report",
    "phonetic": "/rÉªËˆpÉ”Ët/",
    "meaning_vi": "bao cao",
    "topic": "work",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The report is due tomorrow.",
    "target_phonemes": ["r", "Éª", "p", "É”Ë", "t"],
    "common_mistake_tags": ["r_l_confusion", "word_stress_error", "drop_final_consonant"],
    "stress_pattern": "re-PORT"
  },
  {
    "word": "project",
    "phonetic": "/ËˆprÉ’dÊ’ekt/",
    "meaning_vi": "du an",
    "topic": "work",
    "level": "lower_intermediate",
    "difficulty": "medium",
    "sample_sentence": "Our project starts next week.",
    "target_phonemes": ["p", "r", "É’", "dÊ’", "e", "k", "t"],
    "common_mistake_tags": ["consonant_cluster_reduction", "word_stress_error", "drop_final_consonant"],
    "stress_pattern": "PRO-ject"
  },
  {
    "word": "schedule",
    "phonetic": "/ËˆskedÊ’uËl/",
    "meaning_vi": "lich trinh",
    "topic": "work",
    "level": "lower_intermediate",
    "difficulty": "hard",
    "sample_sentence": "Check the schedule before lunch.",
    "target_phonemes": ["s", "k", "e", "dÊ’", "uË", "l"],
    "common_mistake_tags": ["consonant_cluster_reduction", "word_stress_error", "vowel_length_confusion", "r_l_confusion"],
    "stress_pattern": "SCHED-ule"
  },
  {
    "word": "travel",
    "phonetic": "/ËˆtrÃ¦vÉ™l/",
    "meaning_vi": "du lich",
    "topic": "travel",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "I want to travel this summer.",
    "target_phonemes": ["t", "r", "Ã¦", "v", "É™", "l"],
    "common_mistake_tags": ["consonant_cluster_reduction", "ae_e_confusion", "r_l_confusion"],
    "stress_pattern": "TRAV-el"
  },
  {
    "word": "ticket",
    "phonetic": "/ËˆtÉªkÉªt/",
    "meaning_vi": "ve",
    "topic": "travel",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "She buys a train ticket.",
    "target_phonemes": ["t", "Éª", "k", "Éª", "t"],
    "common_mistake_tags": ["short_long_i_confusion", "word_stress_error", "drop_final_consonant"],
    "stress_pattern": "TICK-et"
  },
  {
    "word": "station",
    "phonetic": "/ËˆsteÉªÊƒÉ™n/",
    "meaning_vi": "ga",
    "topic": "travel",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The station is very busy.",
    "target_phonemes": ["s", "t", "eÉª", "Êƒ", "É™", "n"],
    "common_mistake_tags": ["word_stress_error", "sh_ch_confusion"],
    "stress_pattern": "STA-tion"
  },
  {
    "word": "hotel",
    "phonetic": "/hÉ™ÊŠËˆtel/",
    "meaning_vi": "khach san",
    "topic": "travel",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The hotel is near the beach.",
    "target_phonemes": ["h", "É™ÊŠ", "t", "e", "l"],
    "common_mistake_tags": ["word_stress_error", "r_l_confusion", "ae_e_confusion"],
    "stress_pattern": "ho-TEL"
  },
  {
    "word": "map",
    "phonetic": "/mÃ¦p/",
    "meaning_vi": "ban do",
    "topic": "travel",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "Open the map on your phone.",
    "target_phonemes": ["m", "Ã¦", "p"],
    "common_mistake_tags": ["ae_e_confusion", "drop_final_consonant"],
    "stress_pattern": "MAP"
  },
  {
    "word": "think",
    "phonetic": "/Î¸ÉªÅ‹k/",
    "meaning_vi": "nghi",
    "topic": "pronunciation challenges",
    "level": "elementary",
    "difficulty": "hard",
    "sample_sentence": "I think this word is hard.",
    "target_phonemes": ["Î¸", "Éª", "Å‹", "k"],
    "common_mistake_tags": ["th_sound_substitution", "consonant_cluster_reduction", "short_long_i_confusion"],
    "stress_pattern": "THINK"
  },
  {
    "word": "three",
    "phonetic": "/Î¸riË/",
    "meaning_vi": "ba",
    "topic": "pronunciation challenges",
    "level": "beginner",
    "difficulty": "hard",
    "sample_sentence": "I have three books.",
    "target_phonemes": ["Î¸", "r", "iË"],
    "common_mistake_tags": ["th_sound_substitution", "r_l_confusion", "short_long_i_confusion"],
    "stress_pattern": "THREE"
  },
  {
    "word": "this",
    "phonetic": "/Ã°Éªs/",
    "meaning_vi": "day la",
    "topic": "pronunciation challenges",
    "level": "beginner",
    "difficulty": "medium",
    "sample_sentence": "This is my desk.",
    "target_phonemes": ["Ã°", "Éª", "s"],
    "common_mistake_tags": ["th_sound_substitution", "short_long_i_confusion"],
    "stress_pattern": "THIS"
  },
  {
    "word": "mother",
    "phonetic": "/ËˆmÊŒÃ°É™r/",
    "meaning_vi": "me",
    "topic": "daily life",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "My mother is at home.",
    "target_phonemes": ["m", "ÊŒ", "Ã°", "É™", "r"],
    "common_mistake_tags": ["th_sound_substitution", "word_stress_error", "r_l_confusion"],
    "stress_pattern": "MOTH-er"
  },
  {
    "word": "ship",
    "phonetic": "/ÊƒÉªp/",
    "meaning_vi": "con tau",
    "topic": "pronunciation challenges",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The ship leaves tonight.",
    "target_phonemes": ["Êƒ", "Éª", "p"],
    "common_mistake_tags": ["sh_ch_confusion", "short_long_i_confusion", "drop_final_consonant"],
    "stress_pattern": "SHIP"
  },
  {
    "word": "cheap",
    "phonetic": "/tÊƒiËp/",
    "meaning_vi": "re",
    "topic": "pronunciation challenges",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "This bag is cheap.",
    "target_phonemes": ["tÊƒ", "iË", "p"],
    "common_mistake_tags": ["sh_ch_confusion", "short_long_i_confusion", "drop_final_consonant"],
    "stress_pattern": "CHEAP"
  },
  {
    "word": "sheep",
    "phonetic": "/ÊƒiËp/",
    "meaning_vi": "con cuu",
    "topic": "pronunciation challenges",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The sheep are on the hill.",
    "target_phonemes": ["Êƒ", "iË", "p"],
    "common_mistake_tags": ["sh_ch_confusion", "short_long_i_confusion"],
    "stress_pattern": "SHEEP"
  },
  {
    "word": "chip",
    "phonetic": "/tÊƒÉªp/",
    "meaning_vi": "mieng chip",
    "topic": "technology",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The chip is very small.",
    "target_phonemes": ["tÊƒ", "Éª", "p"],
    "common_mistake_tags": ["sh_ch_confusion", "short_long_i_confusion"],
    "stress_pattern": "CHIP"
  },
  {
    "word": "seat",
    "phonetic": "/siËt/",
    "meaning_vi": "cho ngoi",
    "topic": "travel",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "My seat is by the door.",
    "target_phonemes": ["s", "iË", "t"],
    "common_mistake_tags": ["short_long_i_confusion", "drop_final_consonant"],
    "stress_pattern": "SEAT"
  },
  {
    "word": "sit",
    "phonetic": "/sÉªt/",
    "meaning_vi": "ngoi",
    "topic": "daily life",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "Please sit here.",
    "target_phonemes": ["s", "Éª", "t"],
    "common_mistake_tags": ["short_long_i_confusion", "drop_final_consonant"],
    "stress_pattern": "SIT"
  },
  {
    "word": "pen",
    "phonetic": "/pen/",
    "meaning_vi": "but",
    "topic": "school",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "The pen is on the chair.",
    "target_phonemes": ["p", "e", "n"],
    "common_mistake_tags": ["ae_e_confusion", "drop_final_consonant"],
    "stress_pattern": "PEN"
  },
  {
    "word": "bag",
    "phonetic": "/bÃ¦É¡/",
    "meaning_vi": "cai tui",
    "topic": "daily life",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "Her bag is very heavy.",
    "target_phonemes": ["b", "Ã¦", "É¡"],
    "common_mistake_tags": ["ae_e_confusion", "drop_final_consonant"],
    "stress_pattern": "BAG"
  },
  {
    "word": "glass",
    "phonetic": "/É¡lÉ‘Ës/",
    "meaning_vi": "coc thuy tinh",
    "topic": "daily life",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "The glass is on the shelf.",
    "target_phonemes": ["É¡", "l", "É‘Ë", "s"],
    "common_mistake_tags": ["consonant_cluster_reduction", "r_l_confusion", "drop_final_consonant"],
    "stress_pattern": "GLASS"
  },
  {
    "word": "light",
    "phonetic": "/laÉªt/",
    "meaning_vi": "anh sang",
    "topic": "daily life",
    "level": "beginner",
    "difficulty": "easy",
    "sample_sentence": "Turn on the light.",
    "target_phonemes": ["l", "aÉª", "t"],
    "common_mistake_tags": ["r_l_confusion", "drop_final_consonant"],
    "stress_pattern": "LIGHT"
  },
  {
    "word": "right",
    "phonetic": "/raÉªt/",
    "meaning_vi": "dung, ben phai",
    "topic": "pronunciation challenges",
    "level": "elementary",
    "difficulty": "medium",
    "sample_sentence": "Turn right at the corner.",
    "target_phonemes": ["r", "aÉª", "t"],
    "common_mistake_tags": ["r_l_confusion", "drop_final_consonant"],
    "stress_pattern": "RIGHT"
  },
  {
    "word": "road",
    "phonetic": "/rÉ™ÊŠd/",
    "meaning_vi": "con duong",
    "topic": "travel",
    "level": "elementary",
    "difficulty": "easy",
    "sample_sentence": "The road is long and quiet.",
    "target_phonemes": ["r", "É™ÊŠ", "d"],
    "common_mistake_tags": ["r_l_confusion", "drop_final_consonant"],
    "stress_pattern": "ROAD"
  },
  {
    "word": "clock",
    "phonetic": "/klÉ’k/",
    "meaning_vi": "dong ho",
    "topic": "daily life",
    "level": "beginner",
    "difficulty": "medium",
    "sample_sentence": "The clock is above the door.",
    "target_phonemes": ["k", "l", "É’", "k"],
    "common_mistake_tags": ["consonant_cluster_reduction", "r_l_confusion", "final_consonant"],
    "stress_pattern": "CLOCK"
  },
  {
    "word": "train",
    "phonetic": "/treÉªn/",
    "meaning_vi": "tau hoa",
    "topic": "travel",
    "level": "beginner",
    "difficulty": "medium",
    "sample_sentence": "The train leaves at noon.",
    "target_phonemes": ["t", "r", "eÉª", "n"],
    "common_mistake_tags": ["consonant_cluster_reduction", "r_l_confusion"],
    "stress_pattern": "TRAIN"
  },
  {
    "word": "street",
    "phonetic": "/striËt/",
    "meaning_vi": "duong pho",
    "topic": "travel",
    "level": "lower_intermediate",
    "difficulty": "hard",
    "sample_sentence": "The street is crowded tonight.",
    "target_phonemes": ["s", "t", "r", "iË", "t"],
    "common_mistake_tags": ["consonant_cluster_reduction", "r_l_confusion", "short_long_i_confusion", "drop_final_consonant"],
    "stress_pattern": "STREET"
  },
  {
    "word": "world",
    "phonetic": "/wÉœËld/",
    "meaning_vi": "the gioi",
    "topic": "pronunciation challenges",
    "level": "lower_intermediate",
    "difficulty": "hard",
    "sample_sentence": "The world is changing fast.",
    "target_phonemes": ["w", "ÉœË", "l", "d"],
    "common_mistake_tags": ["r_l_confusion", "drop_final_consonant", "vowel_length_confusion"],
    "stress_pattern": "WORLD"
  },
  {
    "word": "architecture",
    "phonetic": "/ËˆÉ‘ËkÉªtektÊƒÉ™r/",
    "meaning_vi": "kien truc",
    "topic": "pronunciation challenges",
    "level": "lower_intermediate",
    "difficulty": "hard",
    "sample_sentence": "Architecture is his favorite subject.",
    "target_phonemes": ["É‘Ë", "k", "Éª", "t", "e", "k", "tÊƒ", "É™", "r"],
    "common_mistake_tags": ["word_stress_error", "sh_ch_confusion", "r_l_confusion", "short_long_i_confusion"],
    "stress_pattern": "AR-chi-tec-ture"
  }
]
$seed_items$::jsonb) as seed(
    word text,
    phonetic text,
    meaning_vi text,
    topic text,
    level text,
    difficulty text,
    sample_sentence text,
    target_phonemes jsonb,
    common_mistake_tags jsonb,
    stress_pattern text
  )
),
normalized_items as (
  select
    btrim(word) as word,
    nullif(btrim(phonetic), '') as phonetic,
    nullif(btrim(meaning_vi), '') as meaning_vi,
    nullif(btrim(topic), '') as topic,
    nullif(btrim(level), '') as level,
    case lower(btrim(difficulty))
      when 'easy' then 1
      when 'medium' then 3
      when 'hard' then 5
      else null
    end::smallint as difficulty,
    nullif(btrim(sample_sentence), '') as sample_sentence,
    target_phonemes,
    common_mistake_tags,
    nullif(btrim(stress_pattern), '') as stress_pattern
  from seed_items_raw
),
inserted_items as (
  insert into public.vocabulary_items (
    word,
    phonetic,
    meaning_vi,
    topic,
    level,
    difficulty,
    sample_sentence,
    target_phonemes,
    common_mistake_tags,
    stress_pattern,
    is_active
  )
  select
    word,
    phonetic,
    meaning_vi,
    topic,
    level,
    difficulty,
    sample_sentence,
    target_phonemes,
    common_mistake_tags,
    stress_pattern,
    true
  from normalized_items
  on conflict (
    lower(btrim(word)),
    coalesce(phonetic, ''),
    coalesce(stress_pattern, '')
  ) do nothing
  returning
    id,
    word,
    phonetic,
    topic,
    common_mistake_tags,
    stress_pattern
),
seed_sets(title, description, topic, level, challenge_tags) as (
  values
    (
      'Final Consonants Practice',
      'Pronunciation drills for final stops, final consonants, and dropped endings.',
      'pronunciation challenges',
      'beginner',
      array['final_consonant', 'drop_final_consonant', 'final_stop_confusion']::text[]
    ),
    (
      'Vowel Length Practice',
      'Pronunciation drills for short and long vowel contrasts.',
      'pronunciation challenges',
      'beginner',
      array['vowel_length_confusion', 'short_long_i_confusion']::text[]
    ),
    (
      'Word Stress Practice',
      'Pronunciation drills for primary stress placement in common words.',
      'pronunciation challenges',
      'elementary',
      array['word_stress_error']::text[]
    ),
    (
      'Consonant Clusters Practice',
      'Pronunciation drills for initial and final consonant clusters.',
      'pronunciation challenges',
      'elementary',
      array['consonant_cluster_reduction']::text[]
    ),
    (
      'Common Vietnamese Learner Challenges',
      'Pronunciation drills for common Vietnamese learner contrasts, including TH sounds, SH/CH, AE/E, and R/L.',
      'pronunciation challenges',
      'elementary',
      array['th_sound_substitution', 'sh_ch_confusion', 'ae_e_confusion', 'r_l_confusion']::text[]
    )
),
inserted_sets as (
  insert into public.vocabulary_sets (
    title,
    description,
    topic,
    level,
    is_public,
    is_active
  )
  select
    ss.title,
    ss.description,
    ss.topic,
    ss.level,
    true,
    true
  from seed_sets as ss
  where not exists (
    select 1
    from public.vocabulary_sets as existing
    where lower(btrim(existing.title)) = lower(btrim(ss.title))
  )
  returning id, title
),
all_seed_sets as (
  select id, title
  from inserted_sets
  union all
  select existing.id, existing.title
  from public.vocabulary_sets as existing
  join seed_sets as ss
    on lower(btrim(existing.title)) = lower(btrim(ss.title))
),
all_seed_items as (
  select
    id,
    word,
    phonetic,
    topic,
    common_mistake_tags,
    stress_pattern
  from inserted_items
  union all
  select
    vi.id,
    vi.word,
    vi.phonetic,
    vi.topic,
    vi.common_mistake_tags,
    vi.stress_pattern
  from normalized_items as ni
  join public.vocabulary_items as vi
    on lower(btrim(vi.word)) = lower(btrim(ni.word))
   and coalesce(vi.phonetic, '') = coalesce(ni.phonetic, '')
   and coalesce(vi.stress_pattern, '') = coalesce(ni.stress_pattern, '')
  where vi.is_active = true
),
set_item_candidates as (
  select
    ass.id as set_id,
    asi.id as item_id,
    row_number() over (
      partition by ass.id
      order by asi.topic, lower(asi.word), asi.stress_pattern
    )::integer as sort_order
  from all_seed_sets as ass
  join seed_sets as ss
    on ss.title = ass.title
  join all_seed_items as asi
    on exists (
      select 1
      from jsonb_array_elements_text(asi.common_mistake_tags) as tag(value)
      where tag.value = any(ss.challenge_tags)
    )
)
insert into public.vocabulary_set_items (
  set_id,
  item_id,
  sort_order
)
select
  set_id,
  item_id,
  sort_order
from set_item_candidates
on conflict on constraint vocabulary_set_items_set_item_unique do update
set sort_order = excluded.sort_order;

commit;
