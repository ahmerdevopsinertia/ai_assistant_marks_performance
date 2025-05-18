import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { ConfigModule } from '@nestjs/config';
@Module({
  imports: [
    ConfigModule.forRoot(), // Add this line
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
