import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { ConfigModule } from '@nestjs/config';
import { HttpModule } from '@nestjs/axios';
import { AnalysisService } from './analysis.service';
import { CacheModule } from '@nestjs/cache-manager';
@Module({
  imports: [
    ConfigModule.forRoot(), // Add this line
    HttpModule.register({
      timeout: 5000,
      maxRedirects: 3,
    }),
    CacheModule.register({
      ttl: 3600, // Cache for 1 hour (in seconds)
      max: 100,  // Maximum number of items in cache
    }),
  ],
  controllers: [AppController],
  providers: [AppService, AnalysisService],
})
export class AppModule {}
