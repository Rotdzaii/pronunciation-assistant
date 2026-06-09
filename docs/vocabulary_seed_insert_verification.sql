-- Vocabulary seed insertion verification queries.
-- Read-only SELECT queries only.
-- Run after docs/vocabulary_seed_insert.sql is manually applied.

-- Count active seed vocabulary items by matching source words and pronunciation metadata.
with expected_items(word, phonetic, stress_pattern) as (
  values
    ('book', '/bʊk/', 'BOOK'),
    ('desk', '/desk/', 'DESK'),
    ('class', '/klæs/', 'CLASS'),
    ('student', '/ˈstjuːdənt/', 'STU-dent'),
    ('teacher', '/ˈtiːtʃər/', 'TEA-cher'),
    ('lesson', '/ˈlesən/', 'LES-son'),
    ('family', '/ˈfæməli/', 'FAM-i-ly'),
    ('breakfast', '/ˈbrekfəst/', 'BREAK-fast'),
    ('kitchen', '/ˈkɪtʃɪn/', 'KITCH-en'),
    ('rice', '/raɪs/', 'RICE'),
    ('milk', '/mɪlk/', 'MILK'),
    ('phone', '/fəʊn/', 'PHONE'),
    ('screen', '/skriːn/', 'SCREEN'),
    ('click', '/klɪk/', 'CLICK'),
    ('chat', '/tʃæt/', 'CHAT'),
    ('video', '/ˈvɪdiəʊ/', 'VID-e-o'),
    ('email', '/ˈiːmeɪl/', 'E-mail'),
    ('report', '/rɪˈpɔːt/', 're-PORT'),
    ('project', '/ˈprɒdʒekt/', 'PRO-ject'),
    ('schedule', '/ˈskedʒuːl/', 'SCHED-ule'),
    ('travel', '/ˈtrævəl/', 'TRAV-el'),
    ('ticket', '/ˈtɪkɪt/', 'TICK-et'),
    ('station', '/ˈsteɪʃən/', 'STA-tion'),
    ('hotel', '/həʊˈtel/', 'ho-TEL'),
    ('map', '/mæp/', 'MAP'),
    ('think', '/θɪŋk/', 'THINK'),
    ('three', '/θriː/', 'THREE'),
    ('this', '/ðɪs/', 'THIS'),
    ('mother', '/ˈmʌðər/', 'MOTH-er'),
    ('ship', '/ʃɪp/', 'SHIP'),
    ('cheap', '/tʃiːp/', 'CHEAP'),
    ('sheep', '/ʃiːp/', 'SHEEP'),
    ('chip', '/tʃɪp/', 'CHIP'),
    ('seat', '/siːt/', 'SEAT'),
    ('sit', '/sɪt/', 'SIT'),
    ('pen', '/pen/', 'PEN'),
    ('bag', '/bæɡ/', 'BAG'),
    ('glass', '/ɡlɑːs/', 'GLASS'),
    ('light', '/laɪt/', 'LIGHT'),
    ('right', '/raɪt/', 'RIGHT'),
    ('road', '/rəʊd/', 'ROAD'),
    ('clock', '/klɒk/', 'CLOCK'),
    ('train', '/treɪn/', 'TRAIN'),
    ('street', '/striːt/', 'STREET'),
    ('world', '/wɜːld/', 'WORLD'),
    ('architecture', '/ˈɑːkɪtektʃər/', 'AR-chi-tec-ture')
)
select
  count(*) as active_seed_item_count,
  46 as expected_seed_item_count
from expected_items as ei
join public.vocabulary_items as vi
  on lower(btrim(vi.word)) = lower(btrim(ei.word))
 and coalesce(vi.phonetic, '') = coalesce(ei.phonetic, '')
 and coalesce(vi.stress_pattern, '') = coalesce(ei.stress_pattern, '')
where vi.is_active = true;

-- Count active public seed vocabulary sets.
select
  count(*) as active_public_seed_set_count,
  5 as expected_seed_set_count
from public.vocabulary_sets
where is_active = true
  and is_public = true
  and title in (
    'Final Consonants Practice',
    'Vowel Length Practice',
    'Word Stress Practice',
    'Consonant Clusters Practice',
    'Common Vietnamese Learner Challenges'
  );

-- Count seed set-item links.
select
  count(*) as seed_set_item_link_count
from public.vocabulary_set_items as vsi
join public.vocabulary_sets as vs
  on vs.id = vsi.set_id
where vs.title in (
  'Final Consonants Practice',
  'Vowel Length Practice',
  'Word Stress Practice',
  'Consonant Clusters Practice',
  'Common Vietnamese Learner Challenges'
);

-- List seed sets and item counts.
select
  vs.title,
  vs.topic,
  vs.level,
  vs.is_public,
  vs.is_active,
  count(vsi.item_id) as item_count
from public.vocabulary_sets as vs
left join public.vocabulary_set_items as vsi
  on vsi.set_id = vs.id
where vs.title in (
  'Final Consonants Practice',
  'Vowel Length Practice',
  'Word Stress Practice',
  'Consonant Clusters Practice',
  'Common Vietnamese Learner Challenges'
)
group by
  vs.title,
  vs.topic,
  vs.level,
  vs.is_public,
  vs.is_active
order by
  vs.title;

-- Sample active seed items by topic.
select
  topic,
  count(*) as item_count,
  array_agg(word order by lower(word)) as sample_words
from public.vocabulary_items
where is_active = true
  and topic in (
    'daily life',
    'pronunciation challenges',
    'school',
    'technology',
    'travel',
    'work'
  )
group by topic
order by topic;

-- Sample active seed items by pronunciation focus tag.
select
  tag.value as focus_tag,
  count(*) as item_count,
  array_agg(vi.word order by lower(vi.word)) as sample_words
from public.vocabulary_items as vi
cross join lateral jsonb_array_elements_text(vi.common_mistake_tags) as tag(value)
where vi.is_active = true
  and tag.value in (
    'final_consonant',
    'drop_final_consonant',
    'final_stop_confusion',
    'vowel_length_confusion',
    'short_long_i_confusion',
    'word_stress_error',
    'consonant_cluster_reduction',
    'th_sound_substitution',
    'sh_ch_confusion',
    'ae_e_confusion',
    'r_l_confusion'
  )
group by tag.value
order by tag.value;
