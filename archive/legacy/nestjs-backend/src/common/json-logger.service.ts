import { LoggerService } from '@nestjs/common';

type LogLevel = 'log' | 'error' | 'warn' | 'debug' | 'verbose' | 'fatal';

export class JsonLoggerService implements LoggerService {
  log(message: any, context?: string) {
    this.writeLog('log', message, context);
  }

  error(message: any, trace?: string, context?: string) {
    this.writeLog('error', message, context, trace);
  }

  warn(message: any, context?: string) {
    this.writeLog('warn', message, context);
  }

  debug(message: any, context?: string) {
    this.writeLog('debug', message, context);
  }

  verbose(message: any, context?: string) {
    this.writeLog('verbose', message, context);
  }

  fatal(message: any, context?: string) {
    this.writeLog('fatal', message, context);
  }

  private writeLog(
    level: LogLevel,
    message: any,
    context?: string,
    trace?: string,
  ) {
    const logPayload = {
      timestamp: new Date().toISOString(),
      level,
      context: context ?? 'Application',
      message: this.normalizeMessage(message),
      trace,
    };

    const output = JSON.stringify(logPayload);

    if (level === 'error' || level === 'fatal') {
      console.error(output);
      return;
    }

    if (level === 'warn') {
      console.warn(output);
      return;
    }

    console.log(output);
  }

  private normalizeMessage(message: any) {
    if (message instanceof Error) {
      return {
        name: message.name,
        message: message.message,
      };
    }

    if (typeof message === 'object') {
      return message;
    }

    return String(message);
  }
}