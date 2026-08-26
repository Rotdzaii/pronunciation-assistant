import React, { useEffect, useRef } from 'react';
import { Animated, Pressable, StyleSheet, Text } from 'react-native';
import { MaterialIcons } from '@expo/vector-icons';
import { colors, typography } from '../../constants/theme';

/** Outer diameter (mobile ≈ 176 dp; web w-48 = 192 px). */
const SIZE = 176;
/** White border thickness matching HTML border-8. */
const BORDER_W = 8;

type Props = {
  isRecording: boolean;
  onPress: () => void;
};

/**
 * Large circular microphone button.
 * Based on the Microphone Button in practice.html.
 *
 * States:
 *  - idle     → bg primaryContainer, label "TAP TO START"
 *  - recording → same bg, label "TAP TO STOP", gentle pulse animation
 */
export default function MicButton({ isRecording, onPress }: Props) {
  const pulse = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (isRecording) {
      const loop = Animated.loop(
        Animated.sequence([
          Animated.timing(pulse, {
            toValue: 1.08,
            duration: 750,
            useNativeDriver: true,
          }),
          Animated.timing(pulse, {
            toValue: 1,
            duration: 750,
            useNativeDriver: true,
          }),
        ]),
      );
      loop.start();
      return () => loop.stop();
    } else {
      pulse.setValue(1);
    }
  }, [isRecording, pulse]);

  return (
    <Animated.View style={{ transform: [{ scale: pulse }] }}>
      <Pressable
        style={({ pressed }) => [
          styles.button,
          pressed && styles.buttonPressed,
        ]}
        onPress={onPress}
        accessibilityRole="button"
        accessibilityLabel={isRecording ? 'Dừng ghi âm' : 'Bắt đầu ghi âm'}
      >
        <MaterialIcons name="mic" size={60} color={colors.onPrimary} />
        <Text style={styles.label}>
          {isRecording ? 'CHẠM ĐỂ DỪNG' : 'CHẠM ĐỂ THU ÂM'}
        </Text>
      </Pressable>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  button: {
    width: SIZE,
    height: SIZE,
    borderRadius: SIZE / 2,
    backgroundColor: colors.micActive,
    borderWidth: BORDER_W,
    borderColor: colors.white,
    alignItems: 'center',
    justifyContent: 'center',
    // Sage-green glow matching micActive
    shadowColor: colors.micActive,
    shadowOffset: { width: 0, height: 16 },
    shadowOpacity: 0.35,
    shadowRadius: 28,
    elevation: 12,
  },
  buttonPressed: {
    transform: [{ scale: 0.92 }],
  },
  label: {
    ...typography.caption,
    color: colors.onPrimary,
    textTransform: 'uppercase',
    letterSpacing: 1.5,
    marginTop: 6,
    opacity: 0.82,
  },
});
