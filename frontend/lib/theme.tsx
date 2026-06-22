import { createContext, PropsWithChildren, useContext, useEffect, useMemo, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

export type ThemeMode = 'light' | 'dark';

export const THEME_MODE_KEY = 'phoenix_theme_mode';

export type ThemeTokens = {
  background: string;
  surface: string;
  surfaceAlt: string;
  sidebar: string;
  card: string;
  cardMuted: string;
  text: string;
  textMuted: string;
  border: string;
  primary: string;
  primaryText: string;
  success: string;
  warning: string;
  danger: string;
  inputBackground: string;
  inputText: string;
  shadow: string;
  softBlue: string;
  softTeal: string;
  softOrange: string;
  softRed: string;
};

export const lightTheme: ThemeTokens = {
  background: '#FAF8FF',
  surface: '#FFFFFF',
  surfaceAlt: '#F8FAFC',
  sidebar: '#F3F3FE',
  card: '#FFFFFF',
  cardMuted: '#F8FAFC',
  text: '#0F172A',
  textMuted: '#64748B',
  border: '#E2E8F0',
  primary: '#2563EB',
  primaryText: '#FFFFFF',
  success: '#22C55E',
  warning: '#F59E0B',
  danger: '#EF4444',
  inputBackground: '#FFFFFF',
  inputText: '#0F172A',
  shadow: '#0F172A',
  softBlue: '#EFF6FF',
  softTeal: '#F0FDFA',
  softOrange: '#FFF7ED',
  softRed: '#FEF2F2',
};

export const darkTheme: ThemeTokens = {
  background: '#0B1020',
  surface: '#111827',
  surfaceAlt: '#0F172A',
  sidebar: '#0F172A',
  card: '#111827',
  cardMuted: '#1E293B',
  text: '#E5E7EB',
  textMuted: '#CBD5E1',
  border: '#334155',
  primary: '#60A5FA',
  primaryText: '#06111F',
  success: '#34D399',
  warning: '#FBBF24',
  danger: '#F87171',
  inputBackground: '#0F172A',
  inputText: '#E5E7EB',
  shadow: '#000000',
  softBlue: '#172554',
  softTeal: '#134E4A',
  softOrange: '#431407',
  softRed: '#450A0A',
};

type ThemeContextValue = {
  mode: ThemeMode;
  theme: ThemeTokens;
  setThemeMode: (mode: ThemeMode) => Promise<void>;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export async function loadThemeMode(): Promise<ThemeMode> {
  const value = await AsyncStorage.getItem(THEME_MODE_KEY);
  return value === 'dark' ? 'dark' : 'light';
}

export async function saveThemeMode(mode: ThemeMode): Promise<void> {
  await AsyncStorage.setItem(THEME_MODE_KEY, mode);
}

export function ThemeProvider({ children }: PropsWithChildren) {
  const [mode, setMode] = useState<ThemeMode>('light');

  useEffect(() => {
    let cancelled = false;

    async function loadSavedTheme() {
      try {
        const savedMode = await loadThemeMode();
        if (!cancelled) {
          setMode(savedMode);
        }
      } catch {
        if (!cancelled) {
          setMode('light');
        }
      }
    }

    loadSavedTheme();

    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo<ThemeContextValue>(() => {
    const theme = mode === 'dark' ? darkTheme : lightTheme;
    return {
      mode,
      theme,
      setThemeMode: async (nextMode: ThemeMode) => {
        setMode(nextMode);
        await saveThemeMode(nextMode);
      },
    };
  }, [mode]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const value = useContext(ThemeContext);
  if (!value) {
    return {
      mode: 'light' as ThemeMode,
      theme: lightTheme,
      setThemeMode: saveThemeMode,
    };
  }
  return value;
}
