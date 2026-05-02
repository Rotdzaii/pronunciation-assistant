import {
  BadRequestException,
  Controller,
  InternalServerErrorException,
  PayloadTooLargeException,
  Post,
  Req,
  UploadedFile,
  UseGuards,
  UseInterceptors,
  UnsupportedMediaTypeException,
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
  constructor(private readonly supabaseService: SupabaseService) {}

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
      .createSignedUrl(filePath, 60 * 60 * 24);

    if (signedUrlError || !signedUrlData?.signedUrl) {
      throw new InternalServerErrorException(
        `Create signed URL failed: ${signedUrlError?.message ?? 'Unknown error'}`,
      );
    }

    return {
      message: 'Audio uploaded successfully',
      uploaded_by: req.user.email,
      app_role: req.user.app_role,
      original_name: file.originalname,
      mime_type: file.mimetype,
      size: file.size,
      storage_path: filePath,
      audio_url: signedUrlData.signedUrl,
    };
  }
}
