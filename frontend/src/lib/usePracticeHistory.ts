import { useCallback, useEffect, useMemo, useState } from 'react';
import type { PracticeHistoryRow } from '../types';
import { supabase } from './supabase';

export function usePracticeHistory(limit = 50) {
  const [history, setHistory] = useState<PracticeHistoryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    const { data, error } = await supabase
      .from('practice_history')
      .select('id,target_word,overall_score,phoneme_details,created_at')
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) {
      setError(error.message);
      setHistory([]);
    } else {
      setHistory((data || []) as PracticeHistoryRow[]);
    }
    setLoading(false);
  }, [limit]);

  useEffect(() => {
    void fetchHistory();
  }, [fetchHistory]);

  const stats = useMemo(() => {
    const latest = history[0];
    const totalAttempts = history.length;

    const daySet = new Set(
      history.map((item) => new Date(item.created_at).toDateString()),
    );
    const days = Array.from(daySet)
      .map((value) => new Date(value))
      .sort((a, b) => b.getTime() - a.getTime());

    let streak = 0;
    if (days.length > 0) {
      let cursor = days[0];
      streak = 1;
      for (let i = 1; i < days.length; i += 1) {
        const next = new Date(cursor);
        next.setDate(cursor.getDate() - 1);
        if (days[i].toDateString() === next.toDateString()) {
          streak += 1;
          cursor = days[i];
        } else {
          break;
        }
      }
    }

    return {
      latestScore: latest?.overall_score ?? null,
      totalAttempts,
      streak,
    };
  }, [history]);

  return {
    history,
    loading,
    error,
    stats,
    refresh: fetchHistory,
  };
}
