import { PracticeService } from './practice.service';
import {
  Body,
  Controller,
  Get,
  Headers,
  HttpCode,
  Param,
  Query,
  Post,
  UploadedFile,
  UseGuards,
  UseInterceptors,
  UnsupportedMediaTypeException,
  PayloadTooLargeException,
  BadRequestException,
  Req,
  InternalServerErrorException,
  UnauthorizedException,
} from '@nestjs/common';
  
import { FileInterceptor } from '@nestjs/platform-express';
import { memoryStorage } from 'multer';
import { randomUUID } from 'crypto';
import { extname } from 'path';

import { SupabaseAuthGuard } from '../auth/supabase-auth.guard';
import { RolesGuard } from '../auth/roles.guard';
import { Roles } from '../auth/roles.decorator';
import { SupabaseService } from '../supabase/supabase.service';

const MAX_FILE_SIZE = 5 * 1024 * 1024;
const ALLOWED_MIME_TYPES = ['audio/wav', 'audio/mpeg', 'audio/mp4'];
const BUCKET_NAME = 'practice-audios';

@Controller('practice')
export class PracticeController {
  constructor(
  private readonly supabaseService: SupabaseService,
  private readonly practiceService: PracticeService,
) {}

  @UseGuards(SupabaseAuthGuard, RolesGuard)
  @Roles('student')
  @Post('upload-audio')
  @UseInterceptors(
    FileInterceptor('audio', {
      storage: memoryStorage(),
      limits: {
        fileSize: MAX_FILE_SIZE,
      },
    }),
  )
  async uploadAudio(
    @UploadedFile() file: Express.Multer.File,
    @Req() req: any,
  ) {
    if (!file) {
      throw new BadRequestException('Audio file is required');
    }

    if (!ALLOWED_MIME_TYPES.includes(file.mimetype)) {
      throw new UnsupportedMediaTypeException(
        'Only audio/wav, audio/mpeg, audio/mp4 are allowed',
      );
    }

    if (file.size > MAX_FILE_SIZE) {
      throw new PayloadTooLargeException('File size must be <= 5MB');
    }

    const fileExtension = extname(file.originalname) || '.bin';
    const filePath = `${req.user.id}/${Date.now()}-${randomUUID()}${fileExtension}`;

    const admin = this.supabaseService.getAdminClient();

    const { error: uploadError } = await admin.storage
      .from(BUCKET_NAME)
      .upload(filePath, file.buffer, {
        contentType: file.mimetype,
        upsert: false,
      });

    if (uploadError) {
      throw new InternalServerErrorException(
        `Upload failed: ${uploadError.message}`,
      );
    }

    const { data: signedUrlData, error: signedUrlError } = await admin.storage
      .from(BUCKET_NAME)
      .createSignedUrl(filePath, 60 * 60 * 24); // 24 giờ

    if (signedUrlError || !signedUrlData?.signedUrl) {
      throw new InternalServerErrorException(
        `Create signed URL failed: ${signedUrlError?.message ?? 'Unknown error'}`,
      );
      
    }

    return {
      message: 'Audio uploaded successfully',
      uploaded_by: req.user.email,
      app_role: req.user.app_role,
      bucket: BUCKET_NAME,
      file_path: filePath,
      audio_url: signedUrlData.signedUrl,
      original_name: file.originalname,
      mime_type: file.mimetype,
      size: file.size,
    };
  }

  @UseGuards(SupabaseAuthGuard, RolesGuard)
@Roles('student')
@HttpCode(202)
@Post('create-job')
async createJob(
  @Body() body: { target_word?: string; audio_url?: string },
  @Req() req: any,
) {
  const { target_word, audio_url } = body;

  if (!target_word || !audio_url) {
    throw new BadRequestException('target_word and audio_url are required');
  }

  return this.practiceService.createJob(
    req.user.id,
    target_word,
    audio_url,
  );
}

  @UseGuards(SupabaseAuthGuard, RolesGuard)
  @Roles('student')
  @Get('history/me')
  async getMyPracticeHistory(
    @Req() req: any,
    @Query('page') page = '1',
    @Query('limit') limit = '10',
  ) {
    return this.practiceService.getMyPracticeHistory(
      req.user.id,
      Number(page),
      Number(limit),
    );
  }

  @UseGuards(SupabaseAuthGuard, RolesGuard)
  @Roles('student')
  @Get(':job_id')
  async getJobStatus(
    @Param('job_id') jobId: string,
    @Req() req: any,
  ) {
    return this.practiceService.getJobStatus(req.user.id, jobId);
  }

  @Post('webhook/ai-result')
  async receiveAiResult(
  @Headers('x-ai-webhook-secret') aiWebhookSecret: string,
    @Body()
    body: {
     job_id?: string;
     status?: 'completed' | 'failed';
     score?: number | null;
     problem_phonemes?: any[];
  },
) {
  const expectedSecret = process.env.AI_WEBHOOK_SECRET;

  if (!expectedSecret) {
    throw new InternalServerErrorException(
      'AI_WEBHOOK_SECRET is not configured',
    );
  }

  if (aiWebhookSecret !== expectedSecret) {
    throw new UnauthorizedException('Invalid AI webhook secret');
  }

  const { job_id, status, score, problem_phonemes } = body;

    if (!job_id || !status) {
      throw new BadRequestException('job_id and status are required');
    }

    if (!['completed', 'failed'].includes(status)) {
      throw new BadRequestException('status must be completed or failed');
    }

    return this.practiceService.updateJobResult(job_id, {
      status,
      score,
      problem_phonemes,
    });
  }
}
