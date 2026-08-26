import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { colors, radius, spacing, typography } from '../../constants/theme';

type Props = {
  icon: React.ReactNode;
  label: string;
  iconBgColor: string;
  /** Informational — caller colors the icon via the icon ReactNode */
  iconColor?: string;
  onPress: () => void;
  disabled?: boolean;
};

/**
 * White card with a circular icon + label below.
 * Based on the Controls Bento Grid in practice.html.
 *
 * Usage:
 *   <ControlButton
 *     icon={<MaterialIcons name="fiber_manual_record" size={24} color={colors.error} />}
 *     iconBgColor={colors.errorContainer}
 *     label="Record"
 *     onPress={handleRecord}
 *   />
 */
export default function ControlButton({
  icon,
  label,
  iconBgColor,
  onPress,
  disabled = false,
}: Props) {
  return (
    <Pressable
      style={({ pressed }) => [
        styles.card,
        disabled && styles.cardDisabled,
        pressed && !disabled && styles.cardPressed,
      ]}
      onPress={onPress}
      disabled={disabled}
    >
      <View style={[styles.iconCircle, { backgroundColor: iconBgColor }]}>
        {icon}
      </View>
      <Text style={styles.label}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
    backgroundColor: colors.white,
    borderRadius: radius.xl,         // 24px — rounded-3xl
    borderWidth: 1,
    borderColor: colors.slate100,
    shadowColor: '#0f172a',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.06,
    shadowRadius: 4,
    elevation: 2,
  },
  cardDisabled: {
    opacity: 0.5,
  },
  cardPressed: {
    opacity: 0.85,
    transform: [{ scale: 0.96 }],
  },
  iconCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.xs,
  },
  label: {
    ...typography.labelBold,
    color: colors.onSurface,
    textAlign: 'center',
  },
});
