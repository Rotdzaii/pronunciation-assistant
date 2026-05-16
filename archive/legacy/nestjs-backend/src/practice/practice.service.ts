import {
  Injectable,
  InternalServerErrorException,
  NotFoundException,
} from '@nestjs/common';
import { randomUUID } from 'crypto';
import { SupabaseService } from '../supabase/supabase.service';

@Injectable()
export class PracticeService {
  constructor(private readonly supabaseService: SupabaseService) {}

  async createJob(studentId: string, targetWord: string, audioUrl: string) {
    const admin = this.supabaseService.getAdminClient();
    const jobId = randomUUID();

    const { error } = await admin.from('practice_history').insert({
      id: jobId,
      student_id: studentId,
      target_word: targetWord,
      audio_url: audioUrl,
      status: 'processing',
      score: null,
      problem_phonemes: [],
    });

    if (error) {
      throw new InternalServerErrorException(
        `Create job failed: ${error.message}`,
      );
    }

    const { error: queueError } = await admin.rpc('enqueue_practice_job', {
      p_job_id: jobId,
      p_student_id: studentId,
      p_target_word: targetWord,
      p_audio_url: audioUrl,
    });

    if (queueError) {
      await admin
        .from('practice_history')
        .update({
          status: 'failed',
          updated_at: new Date().toISOString(),
        })
        .eq('id', jobId);

      throw new InternalServerErrorException(
        `Enqueue practice job failed: ${queueError.message}`,
      );
    }

    return {
      job_id: jobId,
      status: 'processing',
    };
  }

  async getMyPracticeHistory(studentId: string, page = 1, limit = 10) {
    const admin = this.supabaseService.getAdminClient();

    const safePage =
      Number.isFinite(page) && page > 0 ? Math.floor(page) : 1;

    const safeLimit =
      Number.isFinite(limit) && limit > 0
        ? Math.min(Math.floor(limit), 50)
        : 10;

    const from = (safePage - 1) * safeLimit;
    const to = from + safeLimit - 1;

    const { data, error, count } = await admin
      .from('practice_history')
      .select(
        'id, target_word, audio_url, status, score, problem_phonemes, created_at, updated_at',
        { count: 'exact' },
      )
      .eq('student_id', studentId)
      .order('created_at', { ascending: false })
      .range(from, to);

    if (error) {
      throw new InternalServerErrorException(
        `Get practice history failed: ${error.message}`,
      );
    }

    return {
      items: (data ?? []).map((item) => ({
        job_id: item.id,
        target_word: item.target_word,
        audio_url: item.audio_url,
        status: item.status,
        score: item.score,
        problem_phonemes: item.problem_phonemes,
        created_at: item.created_at,
        updated_at: item.updated_at,
      })),
      page: safePage,
      limit: safeLimit,
      total: count ?? 0,
    };
  }

  async getTopProblemPhonemes(limit = 5) {
    const admin = this.supabaseService.getAdminClient();

    const safeLimit =
      Number.isFinite(limit) && limit > 0
        ? Math.min(Math.floor(limit), 20)
        : 5;

    const { data, error } = await admin.rpc('get_top_problem_phonemes', {
      p_limit: safeLimit,
    });

    if (error) {
      throw new InternalServerErrorException(
        `Get top problem phonemes failed: ${error.message}`,
      );
    }

    return {
      items: (data ?? []).map((item) => ({
        phoneme: item.phoneme,
        count: Number(item.error_count),
      })),
      limit: safeLimit,
    };
  }

  async getJobStatus(studentId: string, jobId: string) {
    const admin = this.supabaseService.getAdminClient();

    const { data, error } = await admin
      .from('practice_history')
      .select(
        'id, student_id, target_word, audio_url, status, score, problem_phonemes, created_at, updated_at',
      )
      .eq('id', jobId)
      .eq('student_id', studentId)
      .single();

    if (error || !data) {
      throw new NotFoundException('Job not found');
    }

    return {
      job_id: data.id,
      status: data.status,
      score: data.score,
      problem_phonemes: data.problem_phonemes,
      target_word: data.target_word,
      audio_url: data.audio_url,
      created_at: data.created_at,
      updated_at: data.updated_at,
    };
  }

  async updateJobResult(
    jobId: string,
    payload: {
      status: 'completed' | 'failed';
      score?: number | null;
      problem_phonemes?: any[];
    },
  ) {
    const admin = this.supabaseService.getAdminClient();

    const updateData: any = {
      status: payload.status,
      updated_at: new Date().toISOString(),
    };

    if (payload.status === 'completed') {
      updateData.score = payload.score ?? null;
      updateData.problem_phonemes = payload.problem_phonemes ?? [];
    }

    if (payload.status === 'failed') {
      updateData.score = null;
      updateData.problem_phonemes = [];
    }

    const { data, error } = await admin
      .from('practice_history')
      .update(updateData)
      .eq('id', jobId)
      .select()
      .single();

    if (error || !data) {
      throw new NotFoundException('Job not found');
    }

    return {
      job_id: data.id,
      status: data.status,
      score: data.score,
      problem_phonemes: data.problem_phonemes,
      updated_at: data.updated_at,
    };
  }
}
