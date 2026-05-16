import { Controller, Get, Req, UseGuards } from '@nestjs/common';
import { SupabaseAuthGuard } from './supabase-auth.guard';
import { RolesGuard } from './roles.guard';
import { Roles } from './roles.decorator';

@Controller('auth')
export class AuthController {
  @UseGuards(SupabaseAuthGuard)
  @Get('me')
  me(@Req() req: any) {
    return {
      id: req.user.id,
      email: req.user.email,
      auth_role: req.user.auth_role,
      app_role: req.user.app_role,
    };
  }

  @UseGuards(SupabaseAuthGuard, RolesGuard)
  @Roles('student')
  @Get('student/test')
  studentTest(@Req() req: any) {
    return {
      message: 'Student access granted',
      user: req.user,
    };
  }

  @UseGuards(SupabaseAuthGuard, RolesGuard)
  @Roles('teacher')
  @Get('teacher/test')
  teacherTest(@Req() req: any) {
    return {
      message: 'Teacher access granted',
      user: req.user,
    };
  }
}