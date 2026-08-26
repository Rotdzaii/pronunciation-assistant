import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { colors, radius, spacing, typography } from '../../constants/theme';

type Variant = 'dashed' | 'solid';

type Props = {
  icon: React.ReactNode;
  title: string;
  description: string;
  /**
   * 'dashed' → Hint Box in practice.html (white/semi-transparent, dashed border)
   * 'solid'  → Tip Card in result.html  (tertiaryFixed tinted bg, solid border)
   */
  variant?: Variant;
  /**
   * Override border & icon accent color.
   * Defaults: dashed → primaryContainer, solid → tertiaryFixed
   */
  accentColor?: string;
};

/**
 * Teacher Tip / Learning Tip card.
 *
 * Two visual variants:
 *  - 'dashed' (practice screen): white bg, dashed border in primaryContainer color
 *  - 'solid'  (result screen):   tertiaryFixed tinted bg, solid tertiaryFixed border
 *
 * Note: borderStyle 'dashed' + borderRadius is fully supported on iOS.
 * On Android (RN 0.74) it renders as solid — acceptable for demo.
 */
export default function TipCard({
  icon,
  title,
  description,
  variant = 'dashed',
  accentColor,
}: Props) {
  const isDashed = variant === 'dashed';

  const resolvedBorderColor =
    accentColor ?? (isDashed ? colors.primaryContainer : colors.tertiaryFixed);

  // tertiaryFixed (#ffdea3) at ~30% opacity
  const solidBg = 'rgba(255, 222, 163, 0.30)';

  const titleColor = isDashed ? colors.onSurface : colors.onTertiaryFixed;
  const descColor = isDashed ? colors.onSurfaceVariant : colors.onTertiaryFixedVariant;

  return (
    <View
      style={[
        styles.card,
        {
          backgroundColor: isDashed ? 'rgba(255, 255, 255, 0.40)' : solidBg,
          borderColor: resolvedBorderColor,
          borderStyle: isDashed ? 'dashed' : 'solid',
          borderWidth: isDashed ? 2 : 1.5,
        },
      ]}
    >
      <View style={styles.iconWrap}>{icon}</View>

      <View style={styles.body}>
        <Text style={[styles.title, { color: titleColor }]}>{title}</Text>
        <Text style={[styles.desc, { color: descColor }]}>{description}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.lg,
    borderRadius: radius.xl,    // 24px — rounded-3xl
  },
  iconWrap: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.white,
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 3,
    elevation: 1,
  },
  body: {
    flex: 1,
  },
  title: {
    ...typography.labelBold,
    marginBottom: 4,
  },
  desc: {
    ...typography.caption,
    lineHeight: 18,
  },
});
