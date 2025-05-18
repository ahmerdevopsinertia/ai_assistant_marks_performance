import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import * as dotenv from 'dotenv';

async function bootstrap() {
  dotenv.config(); // Load .env file
  
  const app = await NestFactory.create(AppModule);

  app.setGlobalPrefix('api'); // 👈 Adds /api to all routes

  app.enableCors({
    origin: 'http://localhost:3000', // or your frontend URL
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,
  });
  
  await app.listen(process.env.PORT ?? 3001)
}
bootstrap();
