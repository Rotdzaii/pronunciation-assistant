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

    return {
      job_id: jobId,
      status: 'processing',
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
