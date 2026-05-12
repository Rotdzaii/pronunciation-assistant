import { MaterialCommunityIcons } from '@expo/vector-icons';
import { type Href, useRouter } from 'expo-router';
import { Pressable, StyleSheet, Text, StyleProp, ViewStyle } from 'react-native';

type BackButtonProps = {
  label?: string;
  fallbackHref?: Href<string>;
  style?: StyleProp<ViewStyle>;
};

export function BackButton({ label = 'Quay lại', fallbackHref, style }: BackButtonProps) {
  const router = useRouter();

  const handlePress = () => {
    if (router.canGoBack()) {
      router.back();
      return;
    }

    if (fallbackHref) {
      router.replace(fallbackHref);
    }
  };

  return (
    <Pressable accessibilityRole="button" onPress={handlePress} style={[styles.button, style]}>
      <MaterialCommunityIcons name="arrow-left" size={22} color="#434655" />
      <Text style={styles.label}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    alignSelf: 'flex-start',
    minHeight: 44,
    borderRadius: 999,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 4,
    backgroundColor: 'transparent',
  },
  label: {
    color: '#434655',
    fontSize: 14,
    fontWeight: '800',
  },
});
