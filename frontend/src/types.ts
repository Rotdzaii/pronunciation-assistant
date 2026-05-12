export type Screen =
  | 'ONBOARDING'
  | 'LOGIN'
  | 'DASHBOARD'
  | 'PRACTICE_MENU'
  | 'MISTAKES'
  | 'ANALYTICS'
  | 'PRACTICE_SESSION'
  | 'PROFILE';

export type PhonemeDetail = {
  phoneme: string;
  ipa: string;
  score: number;
};

export type AnalyzeResponse = {
  overall_score: number;
  phoneme_details: PhonemeDetail[];
  ai_thinking?: {
    step_1_vad?: string;
    step_2_stt?: string;
    step_3_alignment?: string;
    step_4_prosody?: string;
    step_5_db?: string;
  };
};

export type PracticeHistoryRow = {
  id?: number;
  target_word: string;
  overall_score: number;
  phoneme_details: PhonemeDetail[];
  created_at: string;
};
