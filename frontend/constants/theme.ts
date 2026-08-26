/**
 * Design tokens extracted from Stitch exports:
 * - practice.html  (M3 color system + spacing/radius/typography)
 * - result.html    (same M3 tokens + emerald/rose/amber semantic additions)
 *
 * Color names map 1-to-1 to the tailwind-config keys in those files.
 * Emerald / rose / amber classes in result.html are mapped to semantic names
 * (success / danger / warning) for clarity in RN code.
 *
 * Font sizes are scaled down ~10-15% from web (mobile baseline).
 */

// ── M3 colour palette (from tailwind-config in practice.html / result.html) ──

export const colors = {
  // Brand / primary
  primary:                '#6844c7',   // bg-primary, text-primary
  primaryContainer:       '#9d7bff',   // bg-primary-container, text-[#9D7BFF]
  primaryFixed:           '#e8ddff',   // bg-primary-fixed (sidebar streak card, tip card)
  onPrimary:              '#ffffff',   // text-on-primary
  onPrimaryFixed:         '#21005e',   // text-on-primary-fixed
  onPrimaryFixedVariant:  '#5028ae',   // text-on-primary-fixed-variant (challenge text, CTA shadow)

  // Secondary / blue
  secondary:              '#005db8',   // text-secondary (badge pill)
  secondaryContainer:     '#4c96fe',   // bg-secondary-container (replay btn, native audio btn, XP text)
  onSecondary:            '#ffffff',   // text-on-secondary

  // Surface / background
  background:             '#fdf7ff',   // bg-background (body)
  surface:                '#fdf7ff',   // same as background in this design
  surfaceLowest:          '#ffffff',   // bg-surface-container-lowest (score card, audio card)
  surfaceContainerLow:    '#f8f1fd',   // bg-surface-container-low
  surfaceContainer:       '#f2ebf7',   // bg-surface-container (avatar bg)
  surfaceContainerHigh:   '#ece6f1',   // bg-surface-container-high (audio compare section)
  surfaceContainerHighest:'#e6e0ec',   // bg-surface-container-highest / bg-surface-variant (stop btn bg)
  surfaceVariant:         '#e6e0ec',   // alias — same value

  // On-surface
  onSurface:              '#1d1a22',   // text-on-surface / text-on-background (default text)
  onSurfaceVariant:       '#494553',   // text-on-surface-variant (secondary text)
  outline:                '#7a7485',   // text-outline (IPA text, caption)
  outlineVariant:         '#cbc3d5',   // border-outline-variant

  // Error
  error:                  '#ba1a1a',   // text-error (record icon)
  errorContainer:         '#ffdad6',   // bg-error-container (record btn bg)

  // Tertiary / amber-ish M3 tokens
  tertiary:               '#7a5900',   // text-tertiary (tip lightbulb icon)
  tertiaryFixed:          '#ffdea3',   // bg-tertiary-fixed (upload btn, tip card bg)
  onTertiaryFixed:        '#261900',   // text-on-tertiary-fixed (upload btn icon, tip card heading)
  onTertiaryFixedVariant: '#5d4200',   // text-on-tertiary-fixed-variant (tip body text)

  // ── Semantic feedback colours (emerald/rose/amber from result.html) ──
  // score-circle CSS: conic-gradient(#10b981 78%, #e2e8f0 0)

  success:       '#10b981',   // emerald-500 — correct word, score circle, toast
  successLight:  '#ecfdf5',   // emerald-50  — score circle outer bg
  successDark:   '#059669',   // emerald-600 — score number text
  successMid:    '#047857',   // emerald-700 — attempt badge text
  successBadge:  '#d1fae5',   // emerald-100 — attempt badge bg

  danger:        '#f43f5e',   // rose-500    — error word highlight
  dangerLight:   '#fff1f2',   // rose-50     — feedback box bg
  dangerBorder:  '#fb7185',   // rose-400    — feedback box left border
  dangerDark:    '#9f1239',   // rose-800    — feedback body text (rose-900 ≈ #881337)

  warning:       '#f59e0b',   // amber-500   — borderline word
  warningLight:  '#fffbeb',   // amber-50    — borderline word bg

  // ── Mic button active colour (olive/sage green) ──
  micActive: '#A9C299',   // custom — replaces primaryContainer on MicButton

  // ── Utility greys (used inline as text-slate-* in both files) ──
  slate50:   '#f8fafc',   // hover bg, bg-slate-50
  slate100:  '#f1f5f9',   // border-slate-100
  slate200:  '#e2e8f0',   // score-circle inactive arc
  slate400:  '#94a3b8',   // secondary labels (streak, XP, IPA subtext)
  slate500:  '#64748b',   // nav inactive icons
  white:     '#ffffff',
} as const;

// ── Spacing (px) — from tailwind-config spacing in both files ──
// base: 4px (not exported, just reference)
export const spacing = {
  xs:  8,    // spacing.xs  = 8px
  sm:  12,   // spacing.sm  = 12px
  md:  16,   // spacing.md  = 16px
  lg:  24,   // spacing.lg  = 24px
  xl:  32,   // spacing.xl  = 32px
  xxl: 48,   // spacing.xxl = 48px
} as const;

// ── Border radius — from tailwind-config borderRadius ──
// DEFAULT = 4px, lg = 8px, xl = 12px, full = 9999px
// Added extra steps used verbatim in HTML (rounded-2xl=16, rounded-3xl=24, rounded-[32px])
export const radius = {
  sm:   8,     // rounded-lg  in tailwind config
  md:   12,    // rounded-xl  in tailwind config
  lg:   16,    // rounded-2xl (used in audio cards, feedback box)
  xl:   24,    // rounded-3xl (controls grid items)
  xxl:  32,    // rounded-[32px] (score card, audio section)
  full: 9999,  // rounded-full (pills, avatar, mic button)
} as const;

// ── Typography — web fontSize scaled down ~12% for mobile ──
// Web → RN:  h1 32→28, h2 24→21, body-lg 18→16, body-md 16→14, label-bold 14→13, caption 13→12
// lineHeight stored as absolute px (RN does not use multiplier).
export const typography = {
  h1: {
    fontSize:   28,
    fontWeight: '700' as const,
    lineHeight: 34,    // web: 32 × 1.2 = 38.4 → scaled ~34
  },
  h2: {
    fontSize:   21,
    fontWeight: '700' as const,
    lineHeight: 27,    // web: 24 × 1.3 = 31.2 → scaled ~27
  },
  bodyLg: {
    fontSize:   16,
    fontWeight: '400' as const,
    lineHeight: 26,    // web: 18 × 1.6 = 28.8 → scaled ~26
  },
  bodyMd: {
    fontSize:   14,
    fontWeight: '400' as const,
    lineHeight: 22,    // web: 16 × 1.6 = 25.6 → scaled ~22
  },
  labelBold: {
    fontSize:     13,
    fontWeight:   '600' as const,
    lineHeight:   16,
    letterSpacing: 0.3,
  },
  caption: {
    fontSize:   12,
    fontWeight: '500' as const,
    lineHeight: 17,    // web: 13 × 1.4 = 18.2 → scaled ~17
  },
} as const;
