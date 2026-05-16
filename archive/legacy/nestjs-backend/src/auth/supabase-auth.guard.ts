import {
  CanActivate,
  ExecutionContext,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { SupabaseService } from '../supabase/supabase.service';

@Injectable()
export class SupabaseAuthGuard implements CanActivate {
  constructor(private readonly supabaseService: SupabaseService) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const authHeader = request.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      throw new UnauthorizedException('Missing or invalid Authorization header');
    }

    const token = authHeader.replace('Bearer ', '').trim();

    const supabase = this.supabaseService.getClient();
    const admin = this.supabaseService.getAdminClient();

    const { data, error } = await supabase.auth.getUser(token);

    if (error || !data.user) {
      throw new UnauthorizedException('Invalid or expired token');
    }

    const user = data.user;

    const { data: profile, error: profileError } = await admin
      .from('profiles')
      .select('email, role')
      .eq('id', user.id)
      .single();

    request.user = {
      id: user.id,
      email: profile?.email ?? user.email,
      auth_role: user.role,
      app_role: profile?.role ?? null,
      profile_error: profileError?.message ?? null,
    };

    return true;
  }
}