import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { colors, typography } from '../../constants/theme';

type Props = {
  icon: React.ReactNode;
  label: string;
  bgColor: string;
  onPress: () => void;
  disabled?: boolean;
};

/**
 * Round icon button with a label below the circle.
 * Used for secondary actions on the practice screen (Replay, Skip).
 * Simpler than ControlButton — no surrounding white card.
 */
export default function CircleIconButton({
  icon,
  label,
  bgColor,
  onPress,
  disabled = false,
}: Props) {
  return (
    <Pressable
      style={({ pressed }) => [
        styles.wrap,
        disabled && styles.disabled,
        pressed && !disabled && styles.pressed,
      ]}
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={label}
    >
      <View style={[styles.circle, { backgroundColor: bgColor }]}>
        {icon}
      </View>
      <Text style={styles.label}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: 'center',
    gap: 8,
  },
  disabled: {
    opacity: 0.45,
  },
  pressed: {
    transform: [{ scale: 0.96 }],
    opacity: 0.88,
  },
  circle: {
    width: 68,
    height: 68,
    borderRadius: 34,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#0f172a',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  label: {
    ...typography.labelBold,
    color: colors.onSurface,
    textAlign: 'center',
  },
});
