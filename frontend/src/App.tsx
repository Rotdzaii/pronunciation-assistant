import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Screen } from './types';
import { Layout } from './components/Layout';
import { OnboardingScreen } from './screens/OnboardingScreen';
import { DashboardScreen } from './screens/DashboardScreen';
import { PracticeMenuScreen } from './screens/PracticeMenuScreen';
import { PracticeSessionScreen } from './screens/PracticeSessionScreen';
import { AnalyticsScreen } from './screens/AnalyticsScreen';
import { MistakesLibraryScreen } from './screens/MistakesLibraryScreen';
import { ProfileScreen } from './screens/ProfileScreen';

export default function App() {
  const [screen, setScreen] = useState<Screen>('ONBOARDING');
  
  const renderScreen = () => {
    switch (screen) {
      case 'ONBOARDING': 
        return <OnboardingScreen onStart={() => setScreen('DASHBOARD')} />;
      case 'DASHBOARD': 
        return <DashboardScreen onPractice={() => setScreen('PRACTICE_MENU')} />;
      case 'PRACTICE_MENU': 
        return <PracticeMenuScreen onStartSession={() => setScreen('PRACTICE_SESSION')} />;
      case 'PRACTICE_SESSION': 
        return <PracticeSessionScreen onComplete={() => setScreen('DASHBOARD')} />;
      case 'MISTAKES': 
        return <MistakesLibraryScreen />;
      case 'ANALYTICS': 
        return <AnalyticsScreen />;
      case 'PROFILE': 
        return <ProfileScreen />;
      default: 
        return <DashboardScreen onPractice={() => setScreen('PRACTICE_MENU')} />;
    }
  };

  // Onboarding should not have navigation
  if (screen === 'ONBOARDING') {
      return renderScreen();
  }

  return (
    <Layout active={screen} onChange={setScreen}>
      <AnimatePresence mode="wait">
        <motion.div
            key={screen}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 10 }}
            transition={{ duration: 0.3 }}
        >
            {renderScreen()}
        </motion.div>
      </AnimatePresence>
    </Layout>
  );
}
